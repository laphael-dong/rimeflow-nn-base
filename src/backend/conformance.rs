//! Shared adapter conformance vocabulary and one-shot Native selection.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::{
    BackendFactory, BackendInitRequest, BackendInstance, BackendKind, ExecutionPlan, Platform,
    ResolvedBackend,
};
use crate::error::{InitFailure, InitializationStage};
use crate::manifest::{ArtifactFormat, ModelManifest};

pub const ADAPTER_CONFORMANCE_SCHEMA_VERSION: u32 = 1;
pub const ADAPTER_CONFORMANCE_SCHEMA_V1: &str =
    include_str!("../../schemas/adapter-conformance.schema.json");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterConformanceCheckKind {
    ManifestIo,
    InitializationTimeout,
    SmokeInference,
    GoldenOutput,
    FaultInjection,
    Diagnostics,
    Performance,
    PackageLoad,
}

impl AdapterConformanceCheckKind {
    pub const ALL: [Self; 8] = [
        Self::ManifestIo,
        Self::InitializationTimeout,
        Self::SmokeInference,
        Self::GoldenOutput,
        Self::FaultInjection,
        Self::Diagnostics,
        Self::Performance,
        Self::PackageLoad,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdapterConformanceStatus {
    Passed,
    BuildVerified,
    Blocked,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ConformanceEvidenceKind {
    RealTarget,
    BuildOnly,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConformanceRunner {
    pub kind: ConformanceEvidenceKind,
    pub target: Platform,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runner_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AdapterConformanceCase {
    pub id: String,
    pub model_id: String,
    pub model_version: String,
    pub target: Platform,
    pub adapter: BackendKind,
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub manifest_sha256: String,
    pub native_initialization_timeout_ms: u64,
}

impl AdapterConformanceCase {
    pub fn validate(&self) -> Result<(), ConformanceReportError> {
        if self.id.is_empty()
            || self.model_id.is_empty()
            || self.model_version.is_empty()
            || self.target.os.is_empty()
            || self.target.arch.is_empty()
            || self.artifact_id.is_empty()
            || self.native_initialization_timeout_ms == 0
        {
            return Err(ConformanceReportError::InvalidCase(
                "case identity, target, artifact, and timeout must be non-empty".to_owned(),
            ));
        }
        if !valid_sha256(&self.artifact_sha256) || !valid_sha256(&self.manifest_sha256) {
            return Err(ConformanceReportError::InvalidCase(
                "artifactSha256 and manifestSha256 must be lowercase SHA-256 values".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AdapterConformanceCheck {
    pub kind: AdapterConformanceCheckKind,
    pub status: AdapterConformanceStatus,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SelectedNativeAdapter {
    pub backend_kind: BackendKind,
    pub platform: Platform,
    pub artifact_id: String,
    pub artifact_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configured_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<String>,
    pub execution_plan: ExecutionPlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
}

impl SelectedNativeAdapter {
    pub fn resolved_backend(
        &self,
        request: &BackendInitRequest,
        initialization_ms: u64,
    ) -> ResolvedBackend {
        ResolvedBackend {
            backend_kind: self.backend_kind,
            platform: self.platform.clone(),
            configured_provider: self.configured_provider.clone(),
            accelerator: self.accelerator.clone(),
            execution_plan: self.execution_plan,
            model_version: request.model_version.clone(),
            artifact_id: self.artifact_id.clone(),
            artifact_sha256: self.artifact_sha256.clone(),
            initialization_ms,
            runtime_version: self.runtime_version.clone(),
        }
    }

    fn matches_resolved(&self, resolved: &ResolvedBackend) -> bool {
        self.backend_kind == resolved.backend_kind
            && self.platform == resolved.platform
            && self.artifact_id == resolved.artifact_id
            && self.artifact_sha256 == resolved.artifact_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum AdapterSelection {
    Ready { selected: SelectedNativeAdapter },
    UseWebFallback { failure: InitFailure },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AdapterConformanceReport {
    pub schema_version: u32,
    pub case: AdapterConformanceCase,
    pub runner: ConformanceRunner,
    pub selection: AdapterSelection,
    pub checks: Vec<AdapterConformanceCheck>,
}

impl AdapterConformanceReport {
    pub fn parse_and_validate(json: &str) -> Result<Self, ConformanceReportError> {
        let report: Self = serde_json::from_str(json)
            .map_err(|error| ConformanceReportError::SchemaInvalid(error.to_string()))?;
        report.validate()?;
        Ok(report)
    }

    pub fn validate(&self) -> Result<(), ConformanceReportError> {
        if self.schema_version != ADAPTER_CONFORMANCE_SCHEMA_VERSION {
            return Err(ConformanceReportError::SchemaInvalid(format!(
                "unsupported adapter conformance schema version {}",
                self.schema_version
            )));
        }
        self.case.validate()?;
        if self.runner.target != self.case.target {
            return Err(ConformanceReportError::RunnerMismatch(
                "runner target does not match the conformance case".to_owned(),
            ));
        }
        match self.runner.kind {
            ConformanceEvidenceKind::RealTarget => {
                if self.runner.runner_id.as_deref().is_none_or(str::is_empty) {
                    return Err(ConformanceReportError::RunnerMismatch(
                        "real-target evidence requires a runnerId".to_owned(),
                    ));
                }
            }
            ConformanceEvidenceKind::BuildOnly | ConformanceEvidenceKind::Unavailable => {
                if self.runner.runner_id.is_some() {
                    return Err(ConformanceReportError::RunnerMismatch(
                        "non-executed evidence must not claim a runnerId".to_owned(),
                    ));
                }
            }
        }

        let mut observed = HashSet::new();
        for check in &self.checks {
            if !observed.insert(check.kind) {
                return Err(ConformanceReportError::CheckSetInvalid(format!(
                    "duplicate {:?} check",
                    check.kind
                )));
            }
            if check.detail.is_empty() {
                return Err(ConformanceReportError::CheckSetInvalid(format!(
                    "{:?} check requires detail",
                    check.kind
                )));
            }
            if matches!(
                check.status,
                AdapterConformanceStatus::Passed | AdapterConformanceStatus::BuildVerified
            ) && check.evidence_path.as_deref().is_none_or(str::is_empty)
            {
                return Err(ConformanceReportError::CheckSetInvalid(format!(
                    "{:?} check requires evidencePath",
                    check.kind
                )));
            }
            if check.status == AdapterConformanceStatus::Passed
                && self.runner.kind != ConformanceEvidenceKind::RealTarget
            {
                return Err(ConformanceReportError::RunnerMismatch(format!(
                    "{:?} cannot pass without real-target evidence",
                    check.kind
                )));
            }
        }
        if AdapterConformanceCheckKind::ALL
            .iter()
            .any(|required| !observed.contains(required))
            || observed.len() != AdapterConformanceCheckKind::ALL.len()
        {
            return Err(ConformanceReportError::CheckSetInvalid(
                "report must contain each required conformance check exactly once".to_owned(),
            ));
        }

        if let AdapterSelection::Ready { selected } = &self.selection {
            if self.runner.kind != ConformanceEvidenceKind::RealTarget {
                return Err(ConformanceReportError::SelectionInvalid(
                    "Ready requires a real target runner".to_owned(),
                ));
            }
            if selected.backend_kind != self.case.adapter
                || selected.platform != self.case.target
                || selected.artifact_id != self.case.artifact_id
                || selected.artifact_sha256 != self.case.artifact_sha256
            {
                return Err(ConformanceReportError::SelectionInvalid(
                    "Ready selection does not match the conformance case".to_owned(),
                ));
            }
            for required in [
                AdapterConformanceCheckKind::ManifestIo,
                AdapterConformanceCheckKind::SmokeInference,
                AdapterConformanceCheckKind::Diagnostics,
            ] {
                let status = self
                    .checks
                    .iter()
                    .find(|check| check.kind == required)
                    .map(|check| check.status);
                if status != Some(AdapterConformanceStatus::Passed) {
                    return Err(ConformanceReportError::SelectionInvalid(format!(
                        "Ready requires {:?} to pass",
                        required
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn overall_status(&self) -> AdapterConformanceStatus {
        if self
            .checks
            .iter()
            .any(|check| check.status == AdapterConformanceStatus::Failed)
        {
            AdapterConformanceStatus::Failed
        } else if self
            .checks
            .iter()
            .any(|check| check.status == AdapterConformanceStatus::Blocked)
        {
            AdapterConformanceStatus::Blocked
        } else if self
            .checks
            .iter()
            .any(|check| check.status == AdapterConformanceStatus::BuildVerified)
        {
            AdapterConformanceStatus::BuildVerified
        } else {
            AdapterConformanceStatus::Passed
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConformanceReportError {
    SchemaInvalid(String),
    InvalidCase(String),
    RunnerMismatch(String),
    CheckSetInvalid(String),
    SelectionInvalid(String),
}

impl std::fmt::Display for ConformanceReportError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemaInvalid(message) => write!(formatter, "invalid report schema: {message}"),
            Self::InvalidCase(message) => write!(formatter, "invalid conformance case: {message}"),
            Self::RunnerMismatch(message) => write!(formatter, "runner mismatch: {message}"),
            Self::CheckSetInvalid(message) => write!(formatter, "invalid check set: {message}"),
            Self::SelectionInvalid(message) => write!(formatter, "invalid selection: {message}"),
        }
    }
}

impl std::error::Error for ConformanceReportError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum CapabilityStatus {
    Ready,
    Unavailable { reason: String },
    Blocked { reason: String },
}

impl CapabilityStatus {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    pub fn blocked(reason: impl Into<String>) -> Self {
        Self::Blocked {
            reason: reason.into(),
        }
    }

    fn unavailable_reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::Unavailable { reason } | Self::Blocked { reason } => Some(reason),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeAdapterCapability {
    pub backend_kind: BackendKind,
    pub target: Platform,
    pub artifact_formats: Vec<ArtifactFormat>,
    pub artifact: CapabilityStatus,
    pub runtime: CapabilityStatus,
    pub device: CapabilityStatus,
    pub smoke: CapabilityStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configured_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<String>,
    pub execution_plan: ExecutionPlan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
}

impl NativeAdapterCapability {
    pub fn ready(
        backend_kind: BackendKind,
        target: Platform,
        artifact_formats: Vec<ArtifactFormat>,
    ) -> Self {
        Self {
            backend_kind,
            target,
            artifact_formats,
            artifact: CapabilityStatus::Ready,
            runtime: CapabilityStatus::Ready,
            device: CapabilityStatus::Ready,
            smoke: CapabilityStatus::Ready,
            configured_provider: None,
            accelerator: None,
            execution_plan: ExecutionPlan::Unknown,
            runtime_version: None,
        }
    }
}

/// Selects one Native adapter from the first request and caches that result.
pub struct PlatformAdapterFactory {
    manifest: ModelManifest,
    capabilities: Vec<NativeAdapterCapability>,
    selection: OnceLock<AdapterSelection>,
    selection_evaluations: AtomicUsize,
}

impl PlatformAdapterFactory {
    pub fn new(manifest: ModelManifest, capabilities: Vec<NativeAdapterCapability>) -> Self {
        Self {
            manifest,
            capabilities,
            selection: OnceLock::new(),
            selection_evaluations: AtomicUsize::new(0),
        }
    }

    pub fn select_once(&self, request: &BackendInitRequest) -> AdapterSelection {
        self.selection
            .get_or_init(|| {
                self.selection_evaluations.fetch_add(1, Ordering::SeqCst);
                self.evaluate(request)
            })
            .clone()
    }

    pub fn selected(&self) -> Option<&AdapterSelection> {
        self.selection.get()
    }

    pub fn selection_evaluation_count(&self) -> usize {
        self.selection_evaluations.load(Ordering::SeqCst)
    }

    fn evaluate(&self, request: &BackendInitRequest) -> AdapterSelection {
        if let Err(error) = self.manifest.validate_semantics() {
            return fallback(
                "manifest_invalid",
                InitializationStage::ManifestParse,
                error.to_string(),
                request,
                None,
            );
        }
        if self.manifest.model.id != request.model_id
            || self.manifest.model.version != request.model_version
        {
            return fallback(
                "manifest_identity_mismatch",
                InitializationStage::ManifestParse,
                "manifest model identity does not match the initialization request",
                request,
                None,
            );
        }
        let artifact = match self
            .manifest
            .select_artifact(&request.artifact_id, &request.target)
        {
            Ok(artifact) => artifact,
            Err(error) => {
                return fallback(
                    "adapter_or_artifact_unavailable",
                    InitializationStage::ArtifactIntegrity,
                    error.to_string(),
                    request,
                    None,
                )
            }
        };
        if artifact.sha256 != request.artifact_sha256 {
            return fallback(
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                "request artifact digest does not match the manifest",
                request,
                None,
            );
        }

        let mut candidates = self.capabilities.iter().filter(|capability| {
            capability.target == request.target
                && capability.artifact_formats.contains(&artifact.format)
                && backend_supports_format(capability.backend_kind, artifact.format)
        });
        let Some(capability) = candidates.next() else {
            return fallback(
                "adapter_or_artifact_unavailable",
                InitializationStage::RuntimeLoad,
                "no Native adapter capability matches the target and artifact format",
                request,
                None,
            );
        };
        if candidates.next().is_some() {
            return fallback(
                "native_adapter_ambiguous",
                InitializationStage::RuntimeLoad,
                "multiple Native adapter capabilities match the first request",
                request,
                None,
            );
        }
        for (status, code, stage, label) in [
            (
                &capability.artifact,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                "artifact",
            ),
            (
                &capability.runtime,
                "native_runtime_unavailable",
                InitializationStage::RuntimeLoad,
                "runtime",
            ),
            (
                &capability.device,
                "native_device_unavailable",
                InitializationStage::DeviceCreate,
                "device",
            ),
            (
                &capability.smoke,
                "native_smoke_failed",
                InitializationStage::SmokeInference,
                "smoke inference",
            ),
        ] {
            if let Some(reason) = status.unavailable_reason() {
                return fallback(
                    code,
                    stage,
                    format!("{label} is not ready: {reason}"),
                    request,
                    Some(capability.backend_kind),
                );
            }
        }

        AdapterSelection::Ready {
            selected: SelectedNativeAdapter {
                backend_kind: capability.backend_kind,
                platform: request.target.clone(),
                artifact_id: artifact.id.clone(),
                artifact_sha256: artifact.sha256.clone(),
                configured_provider: capability.configured_provider.clone(),
                accelerator: capability.accelerator.clone(),
                execution_plan: capability.execution_plan,
                runtime_version: capability.runtime_version.clone(),
            },
        }
    }
}

/// Couples cached platform selection to a backend builder that may run once.
pub struct OneShotNativeAdapterFactory<B, F> {
    selector: PlatformAdapterFactory,
    builder: F,
    build_attempted: AtomicBool,
    _backend: PhantomData<fn() -> B>,
}

impl<B, F> OneShotNativeAdapterFactory<B, F> {
    pub fn new(selector: PlatformAdapterFactory, builder: F) -> Self {
        Self {
            selector,
            builder,
            build_attempted: AtomicBool::new(false),
            _backend: PhantomData,
        }
    }

    pub fn selector(&self) -> &PlatformAdapterFactory {
        &self.selector
    }

    pub fn build_attempt_count(&self) -> usize {
        usize::from(self.build_attempted.load(Ordering::SeqCst))
    }
}

impl<B, F> BackendFactory<B> for OneShotNativeAdapterFactory<B, F>
where
    F: Fn(&BackendInitRequest, &SelectedNativeAdapter) -> Result<BackendInstance<B>, InitFailure>
        + Send
        + Sync,
{
    fn create(&self, request: &BackendInitRequest) -> Result<BackendInstance<B>, InitFailure> {
        let selected = match self.selector.select_once(request) {
            AdapterSelection::Ready { selected } => selected,
            AdapterSelection::UseWebFallback { failure } => return Err(failure),
        };
        if self
            .build_attempted
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(InitFailure::new(
                "native_factory_rebuild_forbidden",
                InitializationStage::RuntimeLoad,
                "the one-shot Native adapter factory has already attempted a build",
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                selected.backend_kind,
            ));
        }
        let instance = (self.builder)(request, &selected)?;
        if !selected.matches_resolved(&instance.resolved) {
            return Err(InitFailure::new(
                "native_factory_diagnostic_mismatch",
                InitializationStage::SmokeInference,
                "built backend diagnostics do not match the fixed adapter selection",
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                selected.backend_kind,
            ));
        }
        Ok(instance)
    }
}

fn fallback(
    code: &'static str,
    stage: InitializationStage,
    message: impl Into<String>,
    request: &BackendInitRequest,
    attempted_backend: Option<BackendKind>,
) -> AdapterSelection {
    let mut failure = InitFailure::new(code, stage, message);
    failure.platform = Some(Box::new(request.target.clone()));
    failure.model_version = Some(request.model_version.clone().into_boxed_str());
    failure.attempted_backend = attempted_backend;
    AdapterSelection::UseWebFallback { failure }
}

fn backend_supports_format(kind: BackendKind, format: ArtifactFormat) -> bool {
    matches!(
        (kind, format),
        (
            BackendKind::LegacyOrt | BackendKind::OpenVino,
            ArtifactFormat::Onnx
        ) | (BackendKind::CoreMl, ArtifactFormat::Coreml)
            | (BackendKind::LiteRtV2, ArtifactFormat::Tflite)
            | (BackendKind::WindowsMl, ArtifactFormat::Windowsml)
            | (BackendKind::MindSporeLite, ArtifactFormat::MindsporeLite)
    )
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
