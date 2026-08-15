//! Minimal public seam used by the Phase 2 test-first contract suite.
//!
//! The actual manifest validator, lifecycle state machine, and adapter factory
//! belong to Phase 4. This module intentionally exposes only the vocabulary
//! required to compile focused red tests and a deterministic fake that always
//! reports the missing behavior by a stable operation name.

use std::fmt;

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
pub enum InitializationStage {
    ManifestParse,
    ArtifactIntegrity,
    RuntimeLoad,
    DeviceCreate,
    ModelCompile,
    IoDiscovery,
    BufferPrepare,
    SmokeInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutBoundary {
    NativeInitialization,
    WebInitialization,
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

/// Deterministic placeholder for contract tests.
///
/// It records calls only for test diagnostics and never implements a Phase 4
/// behavior. Every operation returns the corresponding stable error so a red
/// test cannot accidentally pass because of a fixture or environment failure.
#[derive(Debug, Default)]
pub struct DeterministicRuntimeFake {
    operations: Vec<ContractOperation>,
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
        _manifest_json: &str,
    ) -> Result<(), ContractSeamError> {
        self.not_implemented(ContractOperation::ManifestSchemaValidation)
    }

    pub fn validate_manifest_semantics(
        &mut self,
        _manifest_json: &str,
    ) -> Result<(), ContractSeamError> {
        self.not_implemented(ContractOperation::ManifestSemanticValidation)
    }

    pub fn initialize_native(
        &mut self,
        _request: InitRequest,
    ) -> Result<InitOutcome, ContractSeamError> {
        self.not_implemented(ContractOperation::NativeInitialization)
    }

    pub fn initialize_web(
        &mut self,
        _request: InitRequest,
    ) -> Result<WebInitOutcome, ContractSeamError> {
        self.not_implemented(ContractOperation::WebInitialization)
    }

    pub fn release(&mut self) -> Result<(), ContractSeamError> {
        self.not_implemented(ContractOperation::Release)
    }

    pub fn infer(&mut self) -> Result<(), ContractSeamError> {
        self.not_implemented(ContractOperation::Inference)
    }

    pub fn verify_adapter_conformance(
        &mut self,
        _case: AdapterConformanceCase,
    ) -> Result<AdapterConformanceOutcome, ContractSeamError> {
        self.not_implemented(ContractOperation::AdapterConformance)
    }

    fn not_implemented<T>(&mut self, operation: ContractOperation) -> Result<T, ContractSeamError> {
        self.operations.push(operation);
        Err(ContractSeamError::NotImplemented {
            operation: operation.as_str(),
        })
    }
}
