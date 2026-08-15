//! Compatibility harness for the frozen Phase 2 requirement tests.
//!
//! The test vocabulary remains source-compatible with the original red-test
//! seam, but operations now delegate to the real manifest validator and
//! one-shot lifecycle implementation.

use std::fmt;

use crate::backend::{
    BackendInitRequest as RuntimeInitRequest, BackendInstance, BackendKind as RuntimeBackendKind,
    ExecutionPlan, ModelInput, Platform, RawModelOutput, RawTensor,
    ResolvedBackend as RuntimeResolvedBackend, RuntimeBackend, TensorData,
};
use crate::error::{InferenceError, InitFailure as RuntimeInitFailure};
use crate::lifecycle::{
    InitOutcome as RuntimeInitOutcome, LifecycleSnapshot, RuntimeLifecycle,
    WebInitOutcome as RuntimeWebInitOutcome,
};
use crate::manifest::ModelManifest;

pub use crate::error::{InitializationStage, TimeoutBoundary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    LegacyOrt,
    CoreMl,
    LiteRtV2,
    WindowsMl,
    LinuxOrt,
    MindSporeLite,
    WebOnnx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractOperation {
    ManifestSchemaValidation,
    ManifestSemanticValidation,
    NativeInitialization,
    WebInitialization,
    Release,
    Inference,
    AdapterConformance,
}

impl ContractOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ManifestSchemaValidation => "manifest.schema_validate",
            Self::ManifestSemanticValidation => "manifest.semantic_validate",
            Self::NativeInitialization => "runtime.native_initialize",
            Self::WebInitialization => "runtime.web_initialize",
            Self::Release => "runtime.release",
            Self::Inference => "runtime.infer",
            Self::AdapterConformance => "adapter.conformance",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetPlatform {
    pub os: &'static str,
    pub arch: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitRequest {
    pub target: TargetPlatform,
    pub model_id: &'static str,
    pub injected_fault: Option<InitializationStage>,
    pub timeout: Option<TimeoutBoundary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitFailure {
    pub code: &'static str,
    pub stage: InitializationStage,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBackend {
    pub kind: BackendKind,
    pub target: TargetPlatform,
    pub artifact_id: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitOutcome {
    Ready { resolved: ResolvedBackend },
    UseWebFallback { failure: InitFailure },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebInitOutcome {
    Ready { resolved: ResolvedBackend },
    TerminalFailure { failure: InitFailure },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractSeamError {
    NotImplemented { operation: &'static str },
    ManifestRejected { code: &'static str },
    InferenceFailure { code: &'static str },
}

impl fmt::Display for ContractSeamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented { operation } => write!(formatter, "not_implemented:{operation}"),
            Self::ManifestRejected { code } => write!(formatter, "manifest_rejected:{code}"),
            Self::InferenceFailure { code } => write!(formatter, "inference_failure:{code}"),
        }
    }
}

impl std::error::Error for ContractSeamError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterConformanceCase {
    pub target: TargetPlatform,
    pub adapter: BackendKind,
    pub artifact_id: &'static str,
    pub runtime_evidence_available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterConformanceOutcome {
    Ready { resolved: ResolvedBackend },
    UseWebFallback { failure: InitFailure },
}

struct DeterministicBackend {
    inference_fails: bool,
}

impl RuntimeBackend for DeterministicBackend {
    fn infer(&mut self, _input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        if self.inference_fails {
            return Err(InferenceError::new(
                "inference_failed",
                "deterministic inference fault",
            ));
        }
        Ok(RawModelOutput {
            tensors: vec![RawTensor {
                role: "detections".to_owned(),
                shape: vec![1],
                data: TensorData::F32(vec![0.0]),
            }],
        })
    }
}

#[derive(Default)]
pub struct DeterministicRuntimeFake {
    operations: Vec<ContractOperation>,
    runtime: RuntimeLifecycle<DeterministicBackend>,
}

impl DeterministicRuntimeFake {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn operations(&self) -> &[ContractOperation] {
        &self.operations
    }

    pub fn validate_manifest_schema(
        &mut self,
        manifest_json: &str,
    ) -> Result<(), ContractSeamError> {
        self.operations
            .push(ContractOperation::ManifestSchemaValidation);
        ModelManifest::validate_schema_json(manifest_json)
            .map_err(|error| ContractSeamError::ManifestRejected { code: error.code() })
    }

    pub fn validate_manifest_semantics(
        &mut self,
        manifest_json: &str,
    ) -> Result<(), ContractSeamError> {
        self.operations
            .push(ContractOperation::ManifestSemanticValidation);
        ModelManifest::parse_and_validate(manifest_json)
            .map(|_| ())
            .map_err(|error| ContractSeamError::ManifestRejected { code: error.code() })
    }

    pub fn initialize_native(
        &mut self,
        request: InitRequest,
    ) -> Result<InitOutcome, ContractSeamError> {
        if self.runtime.snapshot() == LifecycleSnapshot::Uninitialized {
            self.operations
                .push(ContractOperation::NativeInitialization);
        }
        let runtime_request = runtime_request(&request, "yolov8n-onnx-fp32");
        let target = request.target.clone();
        let failure = native_failure(&request);
        let factory = move |_: &RuntimeInitRequest| {
            if let Some(failure) = &failure {
                return Err(failure.clone());
            }
            Ok(BackendInstance {
                backend: DeterministicBackend {
                    inference_fails: false,
                },
                resolved: runtime_resolved(
                    RuntimeBackendKind::LinuxOrt,
                    &target,
                    "yolov8n-onnx-fp32",
                ),
            })
        };
        self.runtime
            .initialize_native(&runtime_request, &factory)
            .map(map_native_outcome)
            .map_err(|error| ContractSeamError::InferenceFailure { code: error.code })
    }

    pub fn initialize_web(
        &mut self,
        request: InitRequest,
    ) -> Result<WebInitOutcome, ContractSeamError> {
        self.operations.push(ContractOperation::WebInitialization);
        let runtime_request = runtime_request(&request, "yolov8n-onnx-fp32");
        let target = request.target.clone();
        let timeout = request.timeout;
        let factory = move |_: &RuntimeInitRequest| {
            if timeout == Some(TimeoutBoundary::WebInitialization) {
                return Err(RuntimeInitFailure::new(
                    "web_initialization_timeout",
                    InitializationStage::SmokeInference,
                    "deterministic Web timeout",
                ));
            }
            Ok(BackendInstance {
                backend: DeterministicBackend {
                    inference_fails: false,
                },
                resolved: runtime_resolved(
                    RuntimeBackendKind::WebOnnx,
                    &target,
                    "yolov8n-onnx-fp32",
                ),
            })
        };
        self.runtime
            .initialize_web(&runtime_request, &factory)
            .map(map_web_outcome)
            .map_err(|error| ContractSeamError::InferenceFailure { code: error.code })
    }

    pub fn release(&mut self) -> Result<(), ContractSeamError> {
        self.operations.push(ContractOperation::Release);
        self.runtime
            .release()
            .map_err(|error| ContractSeamError::InferenceFailure { code: error.code })
    }

    pub fn infer(&mut self) -> Result<(), ContractSeamError> {
        self.operations.push(ContractOperation::Inference);
        if self.runtime.snapshot() == LifecycleSnapshot::Uninitialized {
            let request = RuntimeInitRequest {
                target: Platform::current(),
                model_id: "yolov8n".to_owned(),
                model_version: "8.0.0".to_owned(),
                artifact_id: "yolov8n-onnx-fp32".to_owned(),
                artifact_sha256: "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
                    .to_owned(),
            };
            let factory = |request: &RuntimeInitRequest| {
                Ok(BackendInstance {
                    backend: DeterministicBackend {
                        inference_fails: true,
                    },
                    resolved: runtime_resolved(
                        RuntimeBackendKind::LinuxOrt,
                        &TargetPlatform {
                            os: "linux",
                            arch: "x86_64",
                        },
                        request.artifact_id.as_str(),
                    ),
                })
            };
            self.runtime
                .initialize_native(&request, &factory)
                .map_err(|error| ContractSeamError::InferenceFailure { code: error.code })?;
        }
        let result = self.runtime.infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1],
            dtype: crate::backend::DType::F32,
            bytes: 0.0f32.to_le_bytes().to_vec(),
        });
        result
            .map(|_| ())
            .map_err(|error| ContractSeamError::InferenceFailure {
                code: static_code(&error.code),
            })
    }

    pub fn verify_adapter_conformance(
        &mut self,
        case: AdapterConformanceCase,
    ) -> Result<AdapterConformanceOutcome, ContractSeamError> {
        self.operations.push(ContractOperation::AdapterConformance);
        if case.runtime_evidence_available {
            Ok(AdapterConformanceOutcome::Ready {
                resolved: ResolvedBackend {
                    kind: case.adapter,
                    target: case.target,
                    artifact_id: case.artifact_id,
                },
            })
        } else {
            Ok(AdapterConformanceOutcome::UseWebFallback {
                failure: InitFailure {
                    code: "adapter_or_artifact_unavailable",
                    stage: InitializationStage::ArtifactIntegrity,
                },
            })
        }
    }
}

fn runtime_request(request: &InitRequest, artifact_id: &str) -> RuntimeInitRequest {
    RuntimeInitRequest {
        target: Platform::new(request.target.os, request.target.arch),
        model_id: request.model_id.to_owned(),
        model_version: "8.0.0".to_owned(),
        artifact_id: artifact_id.to_owned(),
        artifact_sha256: "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
            .to_owned(),
    }
}

fn runtime_resolved(
    kind: RuntimeBackendKind,
    target: &TargetPlatform,
    artifact_id: &str,
) -> RuntimeResolvedBackend {
    RuntimeResolvedBackend {
        backend_kind: kind,
        platform: Platform::new(target.os, target.arch),
        configured_provider: None,
        accelerator: None,
        execution_plan: ExecutionPlan::Unknown,
        model_version: "8.0.0".to_owned(),
        artifact_id: artifact_id.to_owned(),
        artifact_sha256: "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
            .to_owned(),
        initialization_ms: 0,
        runtime_version: None,
    }
}

fn native_failure(request: &InitRequest) -> Option<RuntimeInitFailure> {
    if request.timeout == Some(TimeoutBoundary::NativeInitialization) {
        return Some(RuntimeInitFailure::new(
            "native_initialization_timeout",
            InitializationStage::RuntimeLoad,
            "deterministic Native timeout",
        ));
    }
    request.injected_fault.map(|stage| {
        RuntimeInitFailure::new(
            "native_initialization_failed",
            stage,
            "deterministic initialization fault",
        )
    })
}

fn map_native_outcome(outcome: RuntimeInitOutcome) -> InitOutcome {
    match outcome {
        RuntimeInitOutcome::Ready { resolved } => InitOutcome::Ready {
            resolved: ResolvedBackend {
                kind: map_backend_kind(resolved.backend_kind),
                target: target_for(&resolved.platform),
                artifact_id: static_artifact_id(&resolved.artifact_id),
            },
        },
        RuntimeInitOutcome::UseWebFallback { failure } => InitOutcome::UseWebFallback {
            failure: InitFailure {
                code: static_code(&failure.code),
                stage: failure.stage,
            },
        },
    }
}

fn map_web_outcome(outcome: RuntimeWebInitOutcome) -> WebInitOutcome {
    match outcome {
        RuntimeWebInitOutcome::Ready { resolved } => WebInitOutcome::Ready {
            resolved: ResolvedBackend {
                kind: map_backend_kind(resolved.backend_kind),
                target: target_for(&resolved.platform),
                artifact_id: static_artifact_id(&resolved.artifact_id),
            },
        },
        RuntimeWebInitOutcome::TerminalFailure { failure } => WebInitOutcome::TerminalFailure {
            failure: InitFailure {
                code: static_code(&failure.code),
                stage: failure.stage,
            },
        },
    }
}

fn map_backend_kind(kind: RuntimeBackendKind) -> BackendKind {
    match kind {
        RuntimeBackendKind::LegacyOrt => BackendKind::LegacyOrt,
        RuntimeBackendKind::CoreMl => BackendKind::CoreMl,
        RuntimeBackendKind::LiteRtV2 => BackendKind::LiteRtV2,
        RuntimeBackendKind::WindowsMl => BackendKind::WindowsMl,
        RuntimeBackendKind::LinuxOrt => BackendKind::LinuxOrt,
        RuntimeBackendKind::MindSporeLite => BackendKind::MindSporeLite,
        RuntimeBackendKind::WebOnnx => BackendKind::WebOnnx,
    }
}

fn target_for(platform: &Platform) -> TargetPlatform {
    match (platform.os.as_str(), platform.arch.as_str()) {
        ("linux", "x86_64") => TargetPlatform {
            os: "linux",
            arch: "x86_64",
        },
        ("macos", "aarch64") => TargetPlatform {
            os: "macos",
            arch: "aarch64",
        },
        ("android", "arm64") => TargetPlatform {
            os: "android",
            arch: "arm64",
        },
        _ => TargetPlatform {
            os: "unknown",
            arch: "unknown",
        },
    }
}

fn static_artifact_id(value: &str) -> &'static str {
    match value {
        "yolov8n-onnx-fp32" => "yolov8n-onnx-fp32",
        "yolov8n-coreml-fp32" => "yolov8n-coreml-fp32",
        "yolov8n-tflite-u8" => "yolov8n-tflite-u8",
        _ => "unknown",
    }
}

fn static_code(value: &str) -> &'static str {
    match value {
        "native_initialization_failed" => "native_initialization_failed",
        "native_initialization_timeout" => "native_initialization_timeout",
        "web_initialization_timeout" => "web_initialization_timeout",
        "adapter_or_artifact_unavailable" => "adapter_or_artifact_unavailable",
        "inference_failed" => "inference_failed",
        "runtime_released" => "runtime_released",
        "runtime_not_ready" => "runtime_not_ready",
        _ => "unexpected_error",
    }
}
