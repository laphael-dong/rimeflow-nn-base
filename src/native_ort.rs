//! Native ONNX Runtime backend via `pykeio/ort` v2.
//!
//! Enabled by `feature = "native"`; per-EP features layer on top
//! (`native-coreml`, `native-directml`, `native-cuda`, `native-tensorrt`,
//! `native-nnapi`, `native-qnn`, `native-openvino`).
//!
//! Model bytes and input shape (`dst_size`) come from the operator crate:
//! the base only knows "there is a square-tile NCHW input"; the operator
//! knows which ONNX file to feed and how big the tile is.
//!
//! ## Current status
//!
//! - **Session lifecycle** — done. Execution providers are stacked in the
//!   order from rules doc §4.4 (CoreML/DirectML → CUDA/TensorRT → NNAPI/QNN
//!   → CPU); the session commits from a byte slice.
//! - **Inference from `wgpu::Buffer`** — CPU-readback path via `map_async`.
//!   `map_async` returns a callback-style Future which we drive with a
//!   synchronous channel + `device.poll(Wait)`. This gives working CPU-EP
//!   inference on all platforms while zero-copy per-EP handoffs (Tier A/B in
//!   rules §4) land in a follow-up.
//! - **Inference from host slice** — done (used by unit tests and any
//!   consumer that already has NCHW host data).
//!
//! ## Follow-up
//!
//! Real Tier A / Tier B zero-copy via `wgpu::Buffer::as_hal::<T>()` per EP:
//! DirectML → `ID3D12Resource`, CoreML → `MTLBuffer.contents()`, CUDA →
//! Vulkan external memory. See rules §5.9 for the handle-extraction skeleton.

#![cfg(all(not(target_arch = "wasm32"), feature = "native"))]

use std::sync::mpsc;

use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use thiserror::Error;

use crate::preprocess::LetterboxParams;

/// Resolved execution provider — recorded so consumers can display
/// "Running on ANE" / "DirectML" etc. in their UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedEp {
    CoreML,
    DirectML,
    Cuda,
    TensorRt,
    Nnapi,
    Qnn,
    OpenVino,
    Cpu,
}

/// ONNX session bound to the best available execution provider.
///
/// The `wgpu_device`/`wgpu_queue` arguments to [`new`](Self::new) are
/// placeholders for future per-EP zero-copy binding (DirectML shared queue,
/// CoreML metal device, CUDA-Vulkan external memory). Currently they are
/// unused; the CPU-readback path only needs the buffer at inference time.
///
/// `dst_size` is the operator's input tile size — the base doesn't hard-code
/// it. `[1, 3, dst_size, dst_size]` is the accepted host shape.
pub struct NativeOrtBackend {
    session:     Session,
    input_name:  String,
    output_name: String,
    dst_size:    u32,
    resolved_ep: ResolvedEp,
}

impl NativeOrtBackend {
    pub fn new(
        model_bytes:  &[u8],
        _wgpu_device: &wgpu::Device,
        _wgpu_queue:  &wgpu::Queue,
        dst_size:     u32,
    ) -> Result<Self, InferError> {
        let (providers, resolved_ep) = build_execution_providers();

        let session = Session::builder()
            .map_err(|e| InferError::OrtBuild(e.to_string()))?
            .with_execution_providers(providers)
            .map_err(|e| InferError::OrtBuild(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferError::OrtBuild(e.to_string()))?
            .commit_from_memory(model_bytes)
            .map_err(|e| InferError::OrtBuild(e.to_string()))?;

        let input_name  = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        Ok(Self { session, input_name, output_name, dst_size, resolved_ep })
    }

    pub fn resolved_ep(&self) -> ResolvedEp { self.resolved_ep }
    pub fn dst_size(&self)    -> u32        { self.dst_size }

    /// Zero/near-zero-copy inference from a wgpu::Buffer produced by
    /// [`crate::preprocess::PreprocessPipeline::dispatch`].
    ///
    /// Current implementation reads the buffer back to host through a staging
    /// buffer and calls [`infer_from_host_slice`](Self::infer_from_host_slice).
    /// Per-EP zero-copy paths are follow-up work; the API contract already
    /// matches the target shape so consumers won't change.
    ///
    /// The staging buffer is allocated per call — cheap on modern DirectX 12
    /// drivers, but callers doing this per frame can cache one externally and
    /// go through `infer_from_host_slice` directly.
    pub fn infer_from_wgpu_buffer(
        &mut self,
        preproc_buffer: &wgpu::Buffer,
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        _letterbox:     LetterboxParams,
    ) -> Result<Vec<f32>, InferError> {
        let size = preproc_buffer.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rimeflow-onnx-base-native-staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rimeflow-onnx-base-native-copy"),
        });
        enc.copy_buffer_to_buffer(preproc_buffer, 0, &staging, 0, size);
        queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()
            .map_err(|_| InferError::MapChannel)??;

        let raw = {
            let bytes = slice.get_mapped_range();
            let nchw: &[f32] = bytemuck::cast_slice(&bytes);
            self.infer_from_host_slice(nchw)?
        };
        staging.unmap();
        Ok(raw)
    }

    /// Inference from a host NCHW `[1, 3, dst_size, dst_size]` f32 slice.
    ///
    /// Used by unit tests, CPU baselines, and consumers that already have
    /// preprocessed data on the host.
    pub fn infer_from_host_slice(&mut self, nchw: &[f32]) -> Result<Vec<f32>, InferError> {
        let dst = self.dst_size as usize;
        let expected = 3 * dst * dst;
        if nchw.len() < expected {
            return Err(InferError::Shape { expected, got: nchw.len() });
        }
        let shape: [i64; 4] = [1, 3, self.dst_size as i64, self.dst_size as i64];
        let tensor = TensorRef::from_array_view((shape, nchw))?;
        let outputs = self.session.run(ort::inputs![&self.input_name => tensor])?;
        let out_ref = outputs
            .get(&self.output_name)
            .ok_or(InferError::OutputMissing)?;
        let (_, out_data) = out_ref.try_extract_tensor::<f32>()?;
        Ok(out_data.to_vec())
    }
}

fn build_execution_providers() -> (Vec<ExecutionProviderDispatch>, ResolvedEp) {
    let mut v: Vec<ExecutionProviderDispatch> = Vec::new();
    #[allow(unused_mut)]
    let mut resolved = ResolvedEp::Cpu;

    #[cfg(feature = "native-coreml")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        v.push(CoreMLExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::CoreML; }
    }
    #[cfg(feature = "native-directml")]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        v.push(DirectMLExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::DirectML; }
    }
    #[cfg(feature = "native-tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;
        v.push(TensorRTExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::TensorRt; }
    }
    #[cfg(feature = "native-cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        v.push(CUDAExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::Cuda; }
    }
    #[cfg(feature = "native-nnapi")]
    {
        use ort::execution_providers::NNAPIExecutionProvider;
        v.push(NNAPIExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::Nnapi; }
    }
    #[cfg(feature = "native-qnn")]
    {
        use ort::execution_providers::QNNExecutionProvider;
        v.push(QNNExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::Qnn; }
    }
    #[cfg(feature = "native-openvino")]
    {
        use ort::execution_providers::OpenVINOExecutionProvider;
        v.push(OpenVINOExecutionProvider::default().build().into());
        if resolved == ResolvedEp::Cpu { resolved = ResolvedEp::OpenVino; }
    }

    // Always append CPU as final fallback.
    use ort::execution_providers::CPUExecutionProvider;
    v.push(CPUExecutionProvider::default().build().into());

    (v, resolved)
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("ort: {0}")]
    Ort(#[from] ort::Error),
    #[error("ort session build: {0}")]
    OrtBuild(String),
    #[error("output tensor missing")]
    OutputMissing,
    #[error("shape mismatch: expected {expected} f32 elements, got {got}")]
    Shape { expected: usize, got: usize },
    #[error("wgpu map_async: {0}")]
    MapAsync(#[from] wgpu::BufferAsyncError),
    #[error("wgpu poll: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("wgpu map_async channel dropped before completion")]
    MapChannel,
}
