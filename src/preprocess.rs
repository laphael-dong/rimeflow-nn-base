//! Preprocess pipeline shared by wasm main-line, native, and canvas-transition paths.
//!
//! Operator crates own their own `shaders/preprocess.wgsl` and pass it in at
//! [`PreprocessPipeline::new`]. All the letterbox math, bind-group layout,
//! and NCHW writeback contract live here.
//!
//! - [`LetterboxParams::compute`] — SINGLE SOURCE for letterbox math. Called
//!   by Rust wgpu preprocess (main-line + native) and mirrored bit-for-bit in
//!   `_computeLetterbox` (JS side of `ort_detect`, see
//!   [`crate::ort_bridge::TEMPLATE_JS`]). Unit tests below pin the invariants;
//!   the JS side must match.
//!
//! - [`PreprocessPipeline`] — the wgpu `ComputePipeline` builder. Both wasm
//!   (WebGPU backend) and native (Metal/Dx12/Vulkan/GL) instantiate it the
//!   same way; the only per-target difference is the `dst_buffer_usage`
//!   chosen by the caller to match the downstream ORT EP (see rules §7.2).

use bytemuck::{Pod, Zeroable};

/// Letterbox transform parameters.
///
/// Both [`PreprocessPipeline::dispatch`] (Rust) and `_computeLetterbox` (JS in
/// `template.js`) must compute these identically. Unit tests below pin the
/// reference values; the generated JS body cross-references the same
/// invariants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxParams {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_size: u32,
    pub scale: f32,
    /// Normalized pad in destination-space (0..1). Multiply by `dst_size` for
    /// pixel-space (needed by the operator crate's postprocess decoder).
    pub pad_x: f32,
    pub pad_y: f32,
}

impl LetterboxParams {
    pub fn compute(src_w: u32, src_h: u32, dst_size: u32) -> Self {
        let dst_f = dst_size as f32;
        let scale = (dst_f / src_w as f32).min(dst_f / src_h as f32);
        let pad_x = (1.0 - (src_w as f32 * scale) / dst_f) / 2.0;
        let pad_y = (1.0 - (src_h as f32 * scale) / dst_f) / 2.0;
        Self {
            src_w,
            src_h,
            dst_size,
            scale,
            pad_x,
            pad_y,
        }
    }

    /// Pad in destination-pixel units, ready for the operator's postprocess.
    pub fn pad_x_px(&self) -> f32 {
        self.pad_x * self.dst_size as f32
    }
    pub fn pad_y_px(&self) -> f32 {
        self.pad_y * self.dst_size as f32
    }
}

/// Uniform buffer layout — matches the base's WGSL `Params` struct contract
/// (see `docs`/rules §16.7.1). Operator crates' WGSL MUST expose the same
/// struct if they want to use this pipeline; deviations require a custom
/// pipeline (SAM2-style multi-scale prep).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PreprocessUniform {
    pub src_w: f32,
    pub src_h: f32,
    pub dst_size: u32,
    pub _pad_a: u32,
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub _pad_b: f32,
}

impl From<LetterboxParams> for PreprocessUniform {
    fn from(lb: LetterboxParams) -> Self {
        Self {
            src_w: lb.src_w as f32,
            src_h: lb.src_h as f32,
            dst_size: lb.dst_size,
            _pad_a: 0,
            scale: lb.scale,
            pad_x: lb.pad_x,
            pad_y: lb.pad_y,
            _pad_b: 0.0,
        }
    }
}

/// Cross-target wgpu preprocess pipeline.
///
/// Instantiate once per session; call [`dispatch`](Self::dispatch) per frame.
/// The output buffer ([`output`](Self::output)) always has NCHW
/// `[1, 3, dst_size, dst_size]` f32 layout.
///
/// The WGSL is passed in by the operator crate at construction time — that's
/// the only knob distinguishing YOLO / SigLIP / SAM2 / … from each other at
/// this layer. The shader MUST bind the standard slots:
///
/// | slot | resource                                               |
/// |------|--------------------------------------------------------|
/// | 0    | `texture_2d<f32>` — scene input                        |
/// | 1    | `sampler`                                              |
/// | 2    | `array<f32>` storage buffer — NCHW output              |
/// | 3    | `Params` uniform buffer (see [`PreprocessUniform`])    |
///
/// See rules doc §16.7 for the reference letterbox prologue.
pub struct PreprocessPipeline {
    pipeline: wgpu::ComputePipeline,
    dst_buffer: wgpu::Buffer,
    uni_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    dst_size: u32,
}

impl PreprocessPipeline {
    /// Create the pipeline.
    ///
    /// `wgsl` is the compute shader source. Operator crates typically pass an
    /// `include_str!("../shaders/preprocess.wgsl")` here — that shader is the
    /// operator's single source of truth (see rules §16.2.2).
    ///
    /// `dst_buffer_usage` is EP-specific:
    ///
    /// | EP                                     | Usage bits                                          |
    /// |----------------------------------------|-----------------------------------------------------|
    /// | Web (onnxruntime-web `fromGpuBuffer`)  | `STORAGE`                                           |
    /// | Web (CPU-slice fallback)               | `STORAGE | MAP_READ | COPY_SRC`                     |
    /// | Native DirectML                        | `STORAGE | COPY_SRC` (share via `as_hal::<Dx12>()`) |
    /// | Native CoreML                          | `STORAGE | MAP_READ` (shared storage mode)          |
    /// | Native CUDA (external memory)          | `STORAGE | COPY_SRC`                                |
    /// | Native NNAPI / CPU                     | `STORAGE | MAP_READ`                                |
    ///
    /// [`Self::default_buffer_usage`] returns a permissive superset suitable
    /// for the CPU-readback path on all EPs; switch to a tighter mask when a
    /// real per-EP zero-copy handoff is wired up.
    pub fn new(
        device: &wgpu::Device,
        dst_size: u32,
        dst_buffer_usage: wgpu::BufferUsages,
        wgsl: &str,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-dst"),
            size: 3u64 * dst_size as u64 * dst_size as u64 * 4,
            usage: dst_buffer_usage,
            mapped_at_creation: false,
        });
        let uni_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-uniform"),
            size: std::mem::size_of::<PreprocessUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipeline,
            dst_buffer,
            uni_buffer,
            sampler,
            dst_size,
        }
    }

    /// Permissive buffer-usage default: `STORAGE | COPY_SRC`. Compatible with:
    /// - [`crate::native_ort::NativeOrtBackend::infer_from_wgpu_buffer`]
    ///   (allocates its own staging buffer for CPU readback).
    /// - Future zero-copy handoffs on DirectML (`as_hal::<Dx12>()` shared
    ///   `ID3D12Resource`) — the STORAGE bit is enough there.
    ///
    /// `MAP_READ` is intentionally excluded — wgpu 29 rejects `MAP_READ` with
    /// any other usage bit (see `Buffer::create` validation). Callers that
    /// need a directly-mapped buffer must allocate a dedicated staging buffer.
    pub fn default_buffer_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    }

    /// Dispatch preprocess for one frame. Caller submits the queue.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene_view: &wgpu::TextureView,
        letterbox: LetterboxParams,
    ) {
        debug_assert_eq!(
            letterbox.dst_size, self.dst_size,
            "PreprocessPipeline::dispatch: letterbox.dst_size mismatch",
        );
        let uni: PreprocessUniform = letterbox.into();
        queue.write_buffer(&self.uni_buffer, 0, bytemuck::bytes_of(&uni));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.pipeline.get_bind_group_layout(0),
            label: Some("rimeflow-onnx-base-preprocess-bg"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uni_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rimeflow-onnx-base-preprocess-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let wg = self.dst_size.div_ceil(16);
        pass.dispatch_workgroups(wg, wg, 1);
    }

    /// Preprocess output buffer — same allocation every frame.
    ///
    /// Callers hand this off to the ORT EP:
    /// - Web:      [`crate::ort_bridge::extract_web_gpu_buffer`] → `web_sys::GpuBuffer`
    /// - DirectML: `output().as_hal::<wgpu_hal::api::Dx12>()` → `ID3D12Resource`
    /// - CoreML:   `output().as_hal::<wgpu_hal::api::Metal>()` → `MTLBuffer`
    /// - CPU:      `output().slice(..).map_async` → `&[f32]`
    pub fn output(&self) -> &wgpu::Buffer {
        &self.dst_buffer
    }
    pub fn dst_size(&self) -> u32 {
        self.dst_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The JS-side `_computeLetterbox` in template.js MUST match these test
    // cases exactly. If any of these move, update the JS too.

    #[test]
    fn letterbox_landscape_1920x1080_640() {
        let lb = LetterboxParams::compute(1920, 1080, 640);
        assert!((lb.scale - (640.0 / 1920.0)).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6);
        assert!((lb.pad_y - 0.21875).abs() < 1e-6);
        assert!(lb.pad_y_px() - 140.0 < 1e-3);
    }

    #[test]
    fn letterbox_portrait_720x1280_640() {
        let lb = LetterboxParams::compute(720, 1280, 640);
        assert!((lb.scale - (640.0 / 1280.0)).abs() < 1e-6);
        assert!(lb.pad_y.abs() < 1e-6);
        assert!(lb.pad_x > 0.0);
    }

    #[test]
    fn letterbox_square_identity() {
        let lb = LetterboxParams::compute(640, 640, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6);
        assert!(lb.pad_y.abs() < 1e-6);
    }

    #[test]
    fn preprocess_uniform_size_matches_wgsl() {
        // `Params` in preprocess.wgsl is 8 * 4 = 32 bytes.
        assert_eq!(std::mem::size_of::<PreprocessUniform>(), 32);
    }
}
