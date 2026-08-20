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

pub mod android_runner;
pub mod backend;
pub mod error;
pub mod lifecycle;
pub mod manifest;

/// Test-first public contract seam for the Phase 2 backend contract.
///
/// This module deliberately contains no manifest parsing, runtime factory, or
/// lifecycle implementation. It gives contract tests a stable, compilable API
/// and a deterministic `not_implemented` fake until Phase 4 owns the real
/// implementation.
pub mod contract_test_seam;

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub mod native_ort;

#[cfg(target_arch = "wasm32")]
pub mod ort_bridge;

pub mod build_helper;

pub use backend::coreml_package_tree_sha256;
#[cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    feature = "openvino-runtime"
))]
pub use backend::{OpenVinoBackend, OpenVinoMetadata};

pub use android_runner::{
    AndroidBundleManifest, AndroidRunnerFile, AndroidRunnerFixture, AndroidRunnerReport,
    AndroidRunnerReportState, ANDROID_RUNNER_BUNDLE_SCHEMA_VERSION,
    ANDROID_RUNNER_REPORT_SCHEMA_VERSION,
};
pub use backend::litert_v2::{
    quantize_f32, LiteRtCompiledRuntime, LiteRtIoPlan, LiteRtRuntimeError, LiteRtTensorBinding,
    LiteRtTensorDescriptor, LiteRtV2Availability, LiteRtV2Backend, LiteRtV2BootstrapError,
    LiteRtV2Diagnostics, VerifiedLiteRtArtifact, LITERT_RUNTIME_VERSION,
    LITERT_RUST_BINDING_VERSION,
};
#[cfg(all(target_os = "android", feature = "litert-v2"))]
pub use backend::litert_v2::{
    AndroidLiteRtAccelerator, AndroidLiteRtV2Backend, AndroidLiteRtV2Factory,
};
pub use backend::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, BackendFactory,
    BackendInitRequest, BackendInstance, BackendKind, CapabilityStatus, ConformanceEvidenceKind,
    ConformanceReportError, ConformanceRunner, CoreMlBackend, CoreMlIoMapping,
    CoreMlPackageIdentity, DType, ExecutionPlan, MindSporeLiteAdapterBuilder,
    MindSporeLiteAvailability, MindSporeLiteBackend, MindSporeLiteBootstrapError,
    MindSporeLiteDiagnostics, MindSporeLiteIoPlan, MindSporeLiteLoadedRuntime,
    MindSporeLiteRuntime, MindSporeLiteRuntimeError, MindSporeLiteRuntimeLoader,
    MindSporeLiteTensorBinding, MindSporeLiteTensorDescriptor, ModelInput, NativeAdapterCapability,
    OneShotNativeAdapterFactory, Platform, PlatformAdapterFactory, RawModelOutput, RawTensor,
    ResolvedBackend, RuntimeBackend, SelectedNativeAdapter, TensorData,
    VerifiedMindSporeLiteArtifact, WindowsMlAdapterConfig, WindowsMlAdapterFactory,
    WindowsMlBackend, WindowsMlMachineReport, WindowsMlRoleBinding, WindowsMlRoleMap,
    WindowsMlRunnerCommand, ADAPTER_CONFORMANCE_SCHEMA_V1, ADAPTER_CONFORMANCE_SCHEMA_VERSION,
    DEFAULT_COREML_INITIALIZATION_TIMEOUT, MINDSPORE_LITE_INFERENCE_TIMEOUT_MS,
    MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS, MINDSPORE_LITE_RUNTIME_VERSION,
    WINDOWS_ML_PACKAGE_VERSION, WINDOWS_ML_RUNNER_SCHEMA_VERSION,
};
pub use error::{InferenceError, InitFailure, InitializationStage, TimeoutBoundary};
pub use lifecycle::{
    InitOutcome, LifecycleError, LifecycleSnapshot, RuntimeLifecycle, WebInitOutcome,
};
pub use manifest::{
    Artifact, ArtifactFormat, ArtifactTarget, Layout, ManifestError, ModelManifest, TensorSpec,
};
pub use preprocess::{LetterboxParams, PreprocessPipeline, PreprocessUniform};

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub use backend::legacy_ort::{LegacyOrtBackend, LegacyOrtMetadata};
#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub use native_ort::{InferError as LegacyInferError, NativeOrtBackend, ResolvedEp};
