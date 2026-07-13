//! # rimeflow-onnx-base
//!
//! Cross-target infrastructure shared by all rimeflow-family ONNX operator
//! crates (`rimeflow-yolov8n`, `rimeflow-siglip`, `rimeflow-sam2`, …).
//! Operator crates depend on this and only own their model-specific files
//! (`postprocess.rs`, `shaders/preprocess.wgsl`, `models/*.onnx`).
//!
//! ## What this crate ships
//!
//! - [`preprocess::PreprocessPipeline`] — cross-target wgpu compute pipeline.
//!   Takes the operator's WGSL as a constructor argument; runs identically on
//!   the WebGPU backend (wasm) and native (Metal / Dx12 / Vulkan / GL).
//! - [`preprocess::LetterboxParams`] — the SINGLE source of letterbox math.
//! - [`native_ort::NativeOrtBackend`] — `pykeio/ort` v2 wrapper with EP
//!   fan-out (CoreML / DirectML / CUDA / TensorRT / NNAPI / QNN / OpenVINO,
//!   CPU last). Feature-gated behind `native` (+ per-EP features).
//! - [`ort_bridge`] — wasm-side helpers (`extract_web_gpu_buffer`,
//!   `get_output_f32`, `get_f64`). The extern block is per-operator (generated
//!   by [`build_helper::generate_extern_block`]).
//! - [`build_helper`] — build-script helper operator crates call from their
//!   own `build.rs` to emit `${OUT_DIR}/ort_bridge_generated.rs`. The base's
//!   `template.js` is embedded at compile time so operator builds do not
//!   need to know where this crate is checked out.
//!
//! ## What operator crates own
//!
//! - `shaders/preprocess.wgsl` — the operator's compute shader (letterbox +
//!   normalize + NCHW writeback). MUST bind slots 0..3 as declared in
//!   [`preprocess::PreprocessPipeline`] docs and MUST define the `Params`
//!   uniform struct compatible with [`preprocess::PreprocessUniform`].
//! - `models/*.onnx` — model weights.
//! - `src/postprocess.rs` — model-specific decode + NMS + result type.
//! - Cargo features: re-export `native`, `native-*`, `model-embedded` verbatim
//!   so downstream consumers list them once.
//! - A newtype or re-export layer that exposes the operator's public API
//!   (see `rimeflow-yolov8n/src/lib.rs` for the canonical example).
//!
//! Full design context: `docs/rimecut-feature-dev-rules.md` §16
//! ("ONNX 基类 Crate 拆分设计").

pub mod preprocess;

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub mod native_ort;

#[cfg(target_arch = "wasm32")]
pub mod ort_bridge;

pub mod build_helper;

pub use preprocess::{LetterboxParams, PreprocessPipeline, PreprocessUniform};
