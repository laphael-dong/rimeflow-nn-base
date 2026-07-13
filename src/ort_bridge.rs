//! wasm-only helpers for consuming the `#[wasm_bindgen(inline_js = ...)]`
//! extern block that operator crates generate with
//! [`crate::build_helper::generate_extern_block`].
//!
//! Operator crates emit their own extern block (one per crate is unavoidable
//! — `wasm_bindgen(inline_js = "…")` needs a string literal at macro
//! expansion time, which excludes generics and constants). The base owns
//! the JS **template**; the operator's `build.rs` substitutes model-specific
//! sentinels (`__PREPROCESS_WGSL__`, `__DST_SIZE__`) and writes the extern
//! block to `${OUT_DIR}/ort_bridge_generated.rs`.
//!
//! The template + helper together guarantee the API surface is verbatim
//! identical across operator crates.

#![cfg(target_arch = "wasm32")]

/// The inline_js template as UTF-8 bytes. [`crate::build_helper`] uses this
/// at build time; runtime code in this crate does not touch it.
///
/// Duplicated in the non-wasm build_helper module via the same
/// `include_str!` — kept as a `const` here so `cargo doc` picks it up.
pub const TEMPLATE_JS: &str = include_str!("../template.js");

/// Extract the underlying `web_sys::GpuBuffer` from a wgpu::Buffer on the
/// WebGPU backend — the wgpu HAL escape hatch (rules §4.5).
///
/// The main-line path (`ort_run_gpu_buffer`) requires this to hand a shared
/// `GPUBuffer` reference to `ort.Tensor.fromGpuBuffer`, achieving Tier A
/// zero-copy on the WebGPU EP.
///
/// # Returns
///
/// - `Some(GpuBuffer)` — the underlying JS-side `GPUBuffer`. Safe to pass to
///   the operator's `ort_run_gpu_buffer` because it lives on the same
///   `GPUDevice` (guaranteed by the `capture_webgpu_device` monkey-patch
///   installed at startup).
/// - `None` — the current wgpu build does not expose the WebGPU HAL escape
///   hatch. Callers MUST fall back to a manual readback + `ort_run_cpu_slice`.
///
/// # wgpu version compatibility
///
/// - wgpu ≤ 24: WebGPU HAL is `pub(crate)`. This function returns `None`.
///   The RimeCut monorepo maintains a tiny patch under `rust/crates/gpu` that
///   unlocks the path.
/// - wgpu ≥ 25 (expected): `Buffer::as_hal::<hal::api::WebGpu>()` will be
///   public. Swap this function body to use it directly.
pub fn extract_web_gpu_buffer(_buf: &wgpu::Buffer) -> Option<web_sys::GpuBuffer> {
    // TODO: once wgpu exposes the WebGPU HAL publicly, implement as:
    //   unsafe {
    //     _buf.as_hal::<wgpu_hal::api::WebGpu>()
    //         .map(|h| h.raw_buffer().clone())
    //   }
    None
}

/// Parse a `Float32Array` field out of the JS dict returned by the operator's
/// `ort_detect`.
pub fn get_output_f32(result: &wasm_bindgen::JsValue, key: &str) -> Option<Vec<f32>> {
    let val = js_sys::Reflect::get(result, &wasm_bindgen::JsValue::from_str(key)).ok()?;
    Some(js_sys::Float32Array::from(val).to_vec())
}

/// Parse a numeric field out of the JS dict returned by the operator's
/// `ort_detect`.
pub fn get_f64(result: &wasm_bindgen::JsValue, key: &str) -> f64 {
    js_sys::Reflect::get(result, &wasm_bindgen::JsValue::from_str(key))
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}
