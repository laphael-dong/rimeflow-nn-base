//! Windows ML adapter that delegates execution to the pinned official package runner.
//!
//! The current Windows ML contract is the Microsoft.WindowsAppSDK.ML 2.1.74
//! distribution. It exposes Microsoft.Windows.AI.MachineLearning catalog APIs
//! and its bundled Microsoft.ML.OnnxRuntime projection. This module does not
//! accept a standalone ORT or DirectML result as Windows ML evidence.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{
    BackendFactory, BackendInitRequest, BackendInstance, BackendKind, DType, ExecutionPlan,
    ModelInput, NativeAdapterCapability, Platform, PlatformAdapterFactory, RawModelOutput,
    RawTensor, ResolvedBackend, RuntimeBackend, SelectedNativeAdapter, TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::{ArtifactFormat, ModelManifest, TensorSpec};

pub const WINDOWS_ML_RUNNER_SCHEMA_VERSION: u32 = 1;
pub const WINDOWS_ML_PACKAGE_VERSION: &str = "2.1.74";

const WINDOWS_ML_SOURCE_PACKAGE: &str = "Microsoft.WindowsAppSDK.ML";
const WINDOWS_ML_RUNTIME_PACKAGE: &str = "Microsoft.Windows.AI.MachineLearning";

/// Command used to invoke the Windows-only runner compiled from
/// `tools/windows-ml-runner`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsMlRunnerCommand {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
}

impl WindowsMlRunnerCommand {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            arguments: Vec::new(),
        }
    }

    pub fn with_arguments<I, A>(program: impl Into<PathBuf>, arguments: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        Self {
            program: program.into(),
            arguments: arguments.into_iter().map(Into::into).collect(),
        }
    }

    /// Creates a command for a framework-dependent published runner DLL.
    pub fn dotnet(assembly: impl Into<PathBuf>) -> Self {
        Self::with_arguments("dotnet", [assembly.into().into_os_string()])
    }
}

/// Runtime configuration for a Windows ML adapter instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowsMlAdapterConfig {
    pub model_path: PathBuf,
    pub runner: WindowsMlRunnerCommand,
    /// This deadline covers the official runner invocation, including its
    /// catalog registration, device selection, session creation, and smoke run.
    pub initialization_timeout: Duration,
}

impl WindowsMlAdapterConfig {
    pub fn new(
        model_path: impl Into<PathBuf>,
        runner: WindowsMlRunnerCommand,
        initialization_timeout: Duration,
    ) -> Self {
        Self {
            model_path: model_path.into(),
            runner,
            initialization_timeout,
        }
    }
}

/// A manifest role mapped to a Windows ML runner tensor binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowsMlRoleBinding {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// One-input, one-or-more-output mapping derived from an artifact manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowsMlRoleMap {
    pub input: WindowsMlRoleBinding,
    pub outputs: Vec<WindowsMlRoleBinding>,
}

impl WindowsMlRoleMap {
    /// Validates the selected manifest artifact and preserves role names or
    /// explicit indexes for the official runner to verify against session metadata.
    pub fn from_manifest(
        manifest: &ModelManifest,
        request: &BackendInitRequest,
    ) -> Result<Self, InitFailure> {
        let artifact = manifest
            .select_artifact(&request.artifact_id, &request.target)
            .map_err(|error| {
                failure(
                    request,
                    "adapter_or_artifact_unavailable",
                    InitializationStage::ArtifactIntegrity,
                    error.to_string(),
                )
            })?;
        if !matches!(
            artifact.format,
            ArtifactFormat::Onnx | ArtifactFormat::Windowsml
        ) {
            return Err(failure(
                request,
                "windows_ml_artifact_format_unsupported",
                InitializationStage::ArtifactIntegrity,
                "Windows ML accepts a canonical ONNX artifact or an explicitly tagged Windows ML artifact",
            ));
        }
        if artifact.sha256 != request.artifact_sha256 {
            return Err(failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                "request artifact digest does not match the manifest",
            ));
        }
        if artifact.inputs.len() != 1 {
            return Err(failure(
                request,
                "windows_ml_multi_input_unsupported",
                InitializationStage::IoDiscovery,
                "the base Windows ML adapter accepts exactly one manifest input role",
            ));
        }

        let input = role_binding(
            manifest.tensors.inputs.as_slice(),
            &artifact.inputs[0],
            request,
        )?;
        let mut output_names = HashSet::new();
        let mut outputs = Vec::with_capacity(artifact.outputs.len());
        for role in &artifact.outputs {
            let output = role_binding(manifest.tensors.outputs.as_slice(), role, request)?;
            let identity = binding_identity(&output);
            if !output_names.insert(identity) {
                return Err(failure(
                    request,
                    "windows_ml_duplicate_output_binding",
                    InitializationStage::IoDiscovery,
                    format!("multiple output roles resolve to {role:?}"),
                ));
            }
            outputs.push(output);
        }
        if outputs.is_empty() {
            return Err(failure(
                request,
                "windows_ml_output_missing",
                InitializationStage::IoDiscovery,
                "the selected Windows ML artifact has no output role",
            ));
        }

        Ok(Self { input, outputs })
    }
}

/// A complete machine report emitted by the official Windows ML runner.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowsMlMachineReport {
    schema_version: u32,
    state: String,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    runtime_executed: bool,
    #[serde(default)]
    failure_stage: Option<String>,
    #[serde(default)]
    windows_ml_api_called: bool,
    #[serde(default)]
    catalog_registration_attempted: bool,
    #[serde(default)]
    catalog_registration_completed: bool,
    #[serde(default)]
    session_created: bool,
    #[serde(default)]
    inference_executed: bool,
    #[serde(default)]
    runtime_introspection_complete: bool,
    #[serde(default)]
    output_published: bool,
    #[serde(default)]
    runtime: Option<WindowsMlRuntimeIdentity>,
    #[serde(default)]
    execution: Option<WindowsMlExecutionIdentity>,
    #[serde(default)]
    error: Option<WindowsMlRunnerError>,
}

impl WindowsMlMachineReport {
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|error| error.to_string())
    }

    pub fn parse_and_validate_runtime_verified(
        json: &str,
        expected_target: &str,
    ) -> Result<Self, String> {
        let report = Self::parse(json)?;
        report.validate_runtime_verified(expected_target)?;
        Ok(report)
    }

    pub fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    pub fn failure_stage(&self) -> Option<&str> {
        self.failure_stage.as_deref()
    }

    pub fn provider_name(&self) -> Option<&str> {
        self.execution
            .as_ref()?
            .selected_device
            .as_ref()
            .map(|device| device.ep_name.as_str())
    }

    pub fn accelerator_name(&self) -> Option<&str> {
        self.execution
            .as_ref()?
            .selected_device
            .as_ref()?
            .hardware
            .as_ref()
            .map(|hardware| hardware.kind.as_str())
    }

    pub fn runtime_version(&self) -> Option<String> {
        let runtime = self.runtime.as_ref()?;
        let ort_version = runtime.ort_version.as_deref()?;
        Some(format!(
            "{WINDOWS_ML_SOURCE_PACKAGE}/{WINDOWS_ML_PACKAGE_VERSION}; Windows ML ORT/{ort_version}"
        ))
    }

    fn error_message(&self) -> Option<&str> {
        self.error.as_ref().map(|error| error.message.as_str())
    }

    fn validate_runtime_verified(&self, expected_target: &str) -> Result<(), String> {
        if self.schema_version != WINDOWS_ML_RUNNER_SCHEMA_VERSION {
            return Err(format!(
                "unsupported Windows ML runner schema version {}",
                self.schema_version
            ));
        }
        if self.state != "runtime-verified" {
            return Err(format!(
                "runner state is {:?}, not runtime-verified",
                self.state
            ));
        }
        if self.target.as_deref() != Some(expected_target) {
            return Err(format!(
                "runner target {:?} does not match {expected_target}",
                self.target
            ));
        }
        if !self.runtime_executed
            || !self.windows_ml_api_called
            || !self.catalog_registration_attempted
            || !self.catalog_registration_completed
            || !self.session_created
            || !self.inference_executed
            || !self.runtime_introspection_complete
            || !self.output_published
        {
            return Err(
                "runner did not complete every required Windows ML lifecycle stage".to_owned(),
            );
        }
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "runner omitted runtime identity".to_owned())?;
        require_package(
            runtime.source_package.as_ref(),
            WINDOWS_ML_SOURCE_PACKAGE,
            "source package",
        )?;
        require_package(
            runtime.runtime_package.as_ref(),
            WINDOWS_ML_RUNTIME_PACKAGE,
            "runtime package",
        )?;
        if runtime.ort_version.as_deref().is_none_or(str::is_empty) {
            return Err("runner omitted the loaded Windows ML ORT version".to_owned());
        }
        if self.provider_name().is_none_or(str::is_empty) {
            return Err("runner omitted the selected Windows ML EP device".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlRuntimeIdentity {
    #[serde(default)]
    source_package: Option<WindowsMlPackageIdentity>,
    #[serde(default)]
    runtime_package: Option<WindowsMlPackageIdentity>,
    #[serde(default)]
    ort_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlPackageIdentity {
    id: String,
    version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlExecutionIdentity {
    #[serde(default)]
    selected_device: Option<WindowsMlDeviceIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlDeviceIdentity {
    ep_name: String,
    #[serde(default)]
    hardware: Option<WindowsMlHardwareIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlHardwareIdentity {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct WindowsMlRunnerError {
    message: String,
}

/// Fixed-selection backend factory for the official Windows ML runner.
pub struct WindowsMlAdapterFactory {
    selector: PlatformAdapterFactory,
    manifest: ModelManifest,
    config: WindowsMlAdapterConfig,
    build_attempted: AtomicBool,
}

impl WindowsMlAdapterFactory {
    pub fn new(manifest: ModelManifest, config: WindowsMlAdapterConfig) -> Self {
        let capabilities = ["x86_64", "aarch64"]
            .into_iter()
            .map(windows_ml_capability)
            .collect();
        Self {
            selector: PlatformAdapterFactory::new(
                selection_manifest_for_windows_ml(&manifest),
                capabilities,
            ),
            manifest,
            config,
            build_attempted: AtomicBool::new(false),
        }
    }

    pub fn selector(&self) -> &PlatformAdapterFactory {
        &self.selector
    }

    pub fn build_attempt_count(&self) -> usize {
        usize::from(self.build_attempted.load(Ordering::SeqCst))
    }
}

impl BackendFactory<WindowsMlBackend> for WindowsMlAdapterFactory {
    fn create(
        &self,
        request: &BackendInitRequest,
    ) -> Result<BackendInstance<WindowsMlBackend>, InitFailure> {
        let selected = match self.selector.select_once(request) {
            super::AdapterSelection::Ready { selected } => selected,
            super::AdapterSelection::UseWebFallback { failure } => return Err(failure),
        };
        if self
            .build_attempted
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(failure(
                request,
                "native_factory_rebuild_forbidden",
                InitializationStage::RuntimeLoad,
                "the one-shot Windows ML adapter factory has already attempted a build",
            ));
        }
        WindowsMlBackend::initialize(&self.manifest, &self.config, request, &selected)
    }
}

/// A backend that runs inference through the official Windows ML runner.
pub struct WindowsMlBackend {
    model_path: PathBuf,
    runner: WindowsMlRunnerCommand,
    roles: WindowsMlRoleMap,
    operation_timeout: Duration,
    resolved: ResolvedBackend,
    last_machine_report: WindowsMlMachineReport,
}

impl WindowsMlBackend {
    pub fn initialize(
        manifest: &ModelManifest,
        config: &WindowsMlAdapterConfig,
        request: &BackendInitRequest,
        selected: &SelectedNativeAdapter,
    ) -> Result<BackendInstance<Self>, InitFailure> {
        ensure_windows_ml_host(request)?;
        let roles = WindowsMlRoleMap::from_manifest(manifest, request)?;
        let artifact = manifest
            .select_artifact(&request.artifact_id, &request.target)
            .map_err(|error| {
                failure(
                    request,
                    "adapter_or_artifact_unavailable",
                    InitializationStage::ArtifactIntegrity,
                    error.to_string(),
                )
            })?;
        let model_path = fs::canonicalize(&config.model_path).map_err(|error| {
            failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                format!("cannot resolve Windows ML model artifact: {error}"),
            )
        })?;
        let model_bytes = fs::read(&model_path).map_err(|error| {
            failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                format!("cannot read Windows ML model artifact: {error}"),
            )
        })?;
        ModelManifest::verify_artifact_bytes(artifact, &model_bytes).map_err(|error| {
            failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                error.to_string(),
            )
        })?;
        if config.initialization_timeout.is_zero() {
            return Err(failure(
                request,
                "native_initialization_timeout",
                InitializationStage::RuntimeLoad,
                "Windows ML initialization timeout must be greater than zero",
            ));
        }

        let started = Instant::now();
        let mut backend = Self {
            model_path,
            runner: config.runner.clone(),
            roles,
            operation_timeout: config.initialization_timeout,
            resolved: selected.resolved_backend(request, 0),
            last_machine_report: unavailable_machine_report(),
        };
        let smoke_input = backend
            .zero_smoke_input()
            .map_err(|error| error.into_init_failure(request))?;
        let (_, report) = backend
            .execute(&smoke_input, "smoke")
            .map_err(|error| error.into_init_failure(request))?;
        let initialization_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        backend.resolved =
            resolved_from_machine_report(selected, request, initialization_ms, &report);
        backend.last_machine_report = report;

        Ok(BackendInstance {
            resolved: backend.resolved.clone(),
            backend,
        })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.resolved
    }

    pub fn role_map(&self) -> &WindowsMlRoleMap {
        &self.roles
    }

    pub fn last_machine_report(&self) -> &WindowsMlMachineReport {
        &self.last_machine_report
    }

    fn zero_smoke_input(&self) -> Result<ModelInput, WindowsMlExecutionError> {
        let byte_len = tensor_byte_len(&self.roles.input.shape)?;
        Ok(ModelInput::Tensor {
            role: self.roles.input.role.clone(),
            shape: self.roles.input.shape.clone(),
            dtype: DType::F32,
            bytes: vec![0; byte_len],
        })
    }

    fn execute(
        &self,
        input: &ModelInput,
        mode: &'static str,
    ) -> Result<(RawModelOutput, WindowsMlMachineReport), WindowsMlExecutionError> {
        let bytes = validate_input(input, &self.roles.input)?;
        let scratch = InvocationScratch::new()?;
        let input_path = scratch.path.join("input-0.f32le");
        fs::write(&input_path, bytes).map_err(|error| {
            WindowsMlExecutionError::new(
                "windows_ml_input_staging_failed",
                InitializationStage::BufferPrepare,
                error.to_string(),
            )
        })?;

        let output_paths: Vec<_> = self
            .roles
            .outputs
            .iter()
            .enumerate()
            .map(|(index, _)| scratch.path.join(format!("output-{index}.f32le")))
            .collect();
        let request_path = scratch.path.join("request.json");
        let report_path = scratch.path.join("report.json");
        let runner_request = WindowsMlRunnerRequest {
            schema_version: WINDOWS_ML_RUNNER_SCHEMA_VERSION,
            mode,
            model_path: path_string(&self.model_path),
            expected_model_sha256: self.resolved.artifact_sha256.clone(),
            inputs: vec![WindowsMlRunnerTensorRequest::from_binding(
                &self.roles.input,
                path_string(&input_path),
            )],
            outputs: self
                .roles
                .outputs
                .iter()
                .zip(&output_paths)
                .map(|(binding, path)| {
                    WindowsMlRunnerTensorRequest::from_binding(binding, path_string(path))
                })
                .collect(),
        };
        let request_json = serde_json::to_vec_pretty(&runner_request).map_err(|error| {
            WindowsMlExecutionError::new(
                "windows_ml_request_serialization_failed",
                InitializationStage::BufferPrepare,
                error.to_string(),
            )
        })?;
        fs::write(&request_path, request_json).map_err(|error| {
            WindowsMlExecutionError::new(
                "windows_ml_request_staging_failed",
                InitializationStage::BufferPrepare,
                error.to_string(),
            )
        })?;

        let status = run_runner(
            &self.runner,
            &request_path,
            &report_path,
            self.operation_timeout,
        )?;
        let report_text = fs::read_to_string(&report_path).map_err(|error| {
            WindowsMlExecutionError::new(
                "windows_ml_runner_report_missing",
                InitializationStage::RuntimeLoad,
                format!("runner exited {status} without a readable report: {error}"),
            )
        })?;
        let report = WindowsMlMachineReport::parse(&report_text).map_err(|error| {
            WindowsMlExecutionError::new(
                "windows_ml_runner_report_invalid",
                InitializationStage::RuntimeLoad,
                error,
            )
        })?;
        if !status.success() {
            let stage = report
                .failure_stage()
                .map(runner_stage)
                .unwrap_or(InitializationStage::RuntimeLoad);
            return Err(WindowsMlExecutionError::new(
                "windows_ml_runner_failed",
                stage,
                format!(
                    "official Windows ML runner exited {status} at {:?}: {}",
                    report.failure_stage(),
                    report.error_message().unwrap_or("no runner error detail")
                ),
            ));
        }
        report
            .validate_runtime_verified(expected_runner_target(&self.resolved.platform)?)
            .map_err(|error| {
                WindowsMlExecutionError::new(
                    "windows_ml_runner_report_invalid",
                    InitializationStage::RuntimeLoad,
                    error,
                )
            })?;

        let tensors = self
            .roles
            .outputs
            .iter()
            .zip(&output_paths)
            .map(|(binding, path)| read_output(binding, path))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((RawModelOutput { tensors }, report))
    }
}

impl RuntimeBackend for WindowsMlBackend {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        let (output, report) = self
            .execute(&input, "infer")
            .map_err(WindowsMlExecutionError::into_inference_error)?;
        self.last_machine_report = report;
        Ok(output)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlRunnerRequest {
    schema_version: u32,
    mode: &'static str,
    model_path: String,
    expected_model_sha256: String,
    inputs: Vec<WindowsMlRunnerTensorRequest>,
    outputs: Vec<WindowsMlRunnerTensorRequest>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WindowsMlRunnerTensorRequest {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<usize>,
    shape: Vec<usize>,
    dtype: DType,
    path: String,
}

impl WindowsMlRunnerTensorRequest {
    fn from_binding(binding: &WindowsMlRoleBinding, path: String) -> Self {
        Self {
            role: binding.role.clone(),
            name: binding.name.clone(),
            index: binding.index,
            shape: binding.shape.clone(),
            dtype: binding.dtype,
            path,
        }
    }
}

#[derive(Debug)]
struct WindowsMlExecutionError {
    code: &'static str,
    stage: InitializationStage,
    message: String,
}

impl WindowsMlExecutionError {
    fn new(code: &'static str, stage: InitializationStage, message: impl Into<String>) -> Self {
        Self {
            code,
            stage,
            message: message.into(),
        }
    }

    fn into_init_failure(self, request: &BackendInitRequest) -> InitFailure {
        let code = if self.code == "windows_ml_runner_timeout" {
            "native_initialization_timeout"
        } else {
            "native_initialization_failed"
        };
        failure(request, code, self.stage, self.message)
    }

    fn into_inference_error(self) -> InferenceError {
        InferenceError::new(self.code, self.message)
    }
}

struct InvocationScratch {
    path: PathBuf,
}

impl InvocationScratch {
    fn new() -> Result<Self, WindowsMlExecutionError> {
        let root = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                WindowsMlExecutionError::new(
                    "windows_ml_temp_directory_failed",
                    InitializationStage::BufferPrepare,
                    error.to_string(),
                )
            })?
            .as_nanos();
        for attempt in 0..32 {
            let path = root.join(format!(
                "rimeflow-windows-ml-{}-{timestamp}-{attempt}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(WindowsMlExecutionError::new(
                        "windows_ml_temp_directory_failed",
                        InitializationStage::BufferPrepare,
                        error.to_string(),
                    ))
                }
            }
        }
        Err(WindowsMlExecutionError::new(
            "windows_ml_temp_directory_failed",
            InitializationStage::BufferPrepare,
            "could not allocate a unique Windows ML invocation directory",
        ))
    }
}

impl Drop for InvocationScratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn windows_ml_capability(arch: &str) -> NativeAdapterCapability {
    let mut capability = NativeAdapterCapability::ready(
        BackendKind::WindowsMl,
        Platform::new("windows", arch),
        vec![ArtifactFormat::Windowsml],
    );
    capability.configured_provider = Some("Windows ML official package".to_owned());
    capability.execution_plan = ExecutionPlan::Unknown;
    capability.runtime_version = Some(format!(
        "{WINDOWS_ML_SOURCE_PACKAGE}/{WINDOWS_ML_PACKAGE_VERSION}"
    ));
    capability
}

fn selection_manifest_for_windows_ml(manifest: &ModelManifest) -> ModelManifest {
    let mut selection_manifest = manifest.clone();
    for artifact in &mut selection_manifest.artifacts {
        if artifact.format == ArtifactFormat::Onnx {
            // The shared selector predates current Windows ML and maps this
            // backend only to `windowsml`. Windows ML 2.1 consumes ONNX
            // directly, so adapt only its private selector view; all runtime
            // validation continues to use the unchanged manifest.
            artifact.format = ArtifactFormat::Windowsml;
        }
    }
    selection_manifest
}

fn role_binding(
    specs: &[TensorSpec],
    role: &str,
    request: &BackendInitRequest,
) -> Result<WindowsMlRoleBinding, InitFailure> {
    let spec = specs.iter().find(|spec| spec.role == role).ok_or_else(|| {
        failure(
            request,
            "manifest_role_invalid",
            InitializationStage::IoDiscovery,
            format!("Windows ML artifact references unknown role {role:?}"),
        )
    })?;
    if spec.dtype != DType::F32 {
        return Err(failure(
            request,
            "windows_ml_dtype_unsupported",
            InitializationStage::BufferPrepare,
            format!(
                "Windows ML base runner requires f32 tensor role {role:?}, found {:?}",
                spec.dtype
            ),
        ));
    }
    let shape = spec
        .shape
        .iter()
        .map(|dimension| usize::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| {
            failure(
                request,
                "static_shape_invalid",
                InitializationStage::IoDiscovery,
                format!("Windows ML role {role:?} has a non-positive static shape"),
            )
        })?;
    if spec.name.as_deref().is_none_or(str::is_empty) && spec.index.is_none() {
        return Err(failure(
            request,
            "manifest_role_invalid",
            InitializationStage::IoDiscovery,
            format!("Windows ML role {role:?} needs a model feature name or index"),
        ));
    }
    Ok(WindowsMlRoleBinding {
        role: role.to_owned(),
        name: spec.name.clone(),
        index: spec.index,
        shape,
        dtype: spec.dtype,
    })
}

fn binding_identity(binding: &WindowsMlRoleBinding) -> String {
    match &binding.name {
        Some(name) => format!("name:{name}"),
        None => format!(
            "index:{}",
            binding.index.expect("manifest validates feature identity")
        ),
    }
}

fn ensure_windows_ml_host(request: &BackendInitRequest) -> Result<(), InitFailure> {
    if !cfg!(feature = "windowsml") {
        return Err(failure(
            request,
            "native_runtime_unavailable",
            InitializationStage::RuntimeLoad,
            "enable the windowsml feature to invoke the official Windows ML runner",
        ));
    }
    if !cfg!(target_os = "windows") {
        return Err(failure(
            request,
            "native_runtime_unavailable",
            InitializationStage::RuntimeLoad,
            "Windows ML Load/Run requires a real Windows target; this host cannot substitute for it",
        ));
    }
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        return Err(failure(
            request,
            "native_runtime_unavailable",
            InitializationStage::RuntimeLoad,
            "Windows ML adapter supports only Windows x86_64 and aarch64 targets",
        ));
    }
    if request.target != Platform::current() {
        return Err(failure(
            request,
            "windows_ml_target_mismatch",
            InitializationStage::RuntimeLoad,
            format!(
                "requested {}/{} but this Windows binary targets {}/{}",
                request.target.os,
                request.target.arch,
                Platform::current().os,
                Platform::current().arch
            ),
        ));
    }
    Ok(())
}

fn validate_input<'a>(
    input: &'a ModelInput,
    expected: &WindowsMlRoleBinding,
) -> Result<&'a [u8], WindowsMlExecutionError> {
    input.validate().map_err(|error| {
        WindowsMlExecutionError::new(
            "invalid_input",
            InitializationStage::BufferPrepare,
            error.to_string(),
        )
    })?;
    let ModelInput::Tensor {
        role,
        shape,
        dtype,
        bytes,
    } = input
    else {
        return Err(WindowsMlExecutionError::new(
            "unsupported_input",
            InitializationStage::BufferPrepare,
            "Windows ML requires a manifest-mapped f32 tensor input",
        ));
    };
    if role != &expected.role || shape != &expected.shape {
        return Err(WindowsMlExecutionError::new(
            "windows_ml_input_role_or_shape_mismatch",
            InitializationStage::BufferPrepare,
            format!(
                "expected role {:?} and shape {:?}, got role {:?} and shape {:?}",
                expected.role, expected.shape, role, shape
            ),
        ));
    }
    if *dtype != DType::F32 {
        return Err(WindowsMlExecutionError::new(
            "windows_ml_dtype_unsupported",
            InitializationStage::BufferPrepare,
            "Windows ML base runner accepts only f32 inputs",
        ));
    }
    for bytes in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(bytes.try_into().expect("exact f32 chunks"));
        if !value.is_finite() {
            return Err(WindowsMlExecutionError::new(
                "windows_ml_input_non_finite",
                InitializationStage::BufferPrepare,
                "Windows ML input contains a non-finite f32 value",
            ));
        }
    }
    Ok(bytes)
}

fn read_output(
    binding: &WindowsMlRoleBinding,
    path: &Path,
) -> Result<RawTensor, WindowsMlExecutionError> {
    let bytes = fs::read(path).map_err(|error| {
        WindowsMlExecutionError::new(
            "windows_ml_output_missing",
            InitializationStage::SmokeInference,
            format!("{}: {error}", path.display()),
        )
    })?;
    let expected_len = tensor_byte_len(&binding.shape)?;
    if bytes.len() != expected_len {
        return Err(WindowsMlExecutionError::new(
            "windows_ml_output_invalid",
            InitializationStage::SmokeInference,
            format!(
                "role {:?} expected {expected_len} output bytes, got {}",
                binding.role,
                bytes.len()
            ),
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact f32 chunks")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(WindowsMlExecutionError::new(
            "windows_ml_output_non_finite",
            InitializationStage::SmokeInference,
            format!("role {:?} contains a non-finite f32 output", binding.role),
        ));
    }
    Ok(RawTensor {
        role: binding.role.clone(),
        shape: binding.shape.clone(),
        data: TensorData::F32(values),
    })
}

fn tensor_byte_len(shape: &[usize]) -> Result<usize, WindowsMlExecutionError> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            WindowsMlExecutionError::new(
                "input_size_overflow",
                InitializationStage::BufferPrepare,
                "Windows ML tensor shape overflows its f32 byte length",
            )
        })
}

fn run_runner(
    runner: &WindowsMlRunnerCommand,
    request_path: &Path,
    report_path: &Path,
    timeout: Duration,
) -> Result<ExitStatus, WindowsMlExecutionError> {
    let mut command = Command::new(&runner.program);
    command
        .args(&runner.arguments)
        .arg("--request")
        .arg(request_path)
        .arg("--report")
        .arg(report_path);
    let mut child = command.spawn().map_err(|error| {
        WindowsMlExecutionError::new(
            "windows_ml_runner_spawn_failed",
            InitializationStage::RuntimeLoad,
            format!("{}: {error}", runner.program.display()),
        )
    })?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(WindowsMlExecutionError::new(
                    "windows_ml_runner_timeout",
                    InitializationStage::RuntimeLoad,
                    format!(
                        "official Windows ML runner exceeded the {} ms deadline",
                        timeout.as_millis()
                    ),
                ));
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(error) => {
                return Err(WindowsMlExecutionError::new(
                    "windows_ml_runner_wait_failed",
                    InitializationStage::RuntimeLoad,
                    error.to_string(),
                ))
            }
        }
    }
}

fn expected_runner_target(platform: &Platform) -> Result<&'static str, WindowsMlExecutionError> {
    match (platform.os.as_str(), platform.arch.as_str()) {
        ("windows", "x86_64") => Ok("win-x64"),
        ("windows", "aarch64") => Ok("win-arm64"),
        _ => Err(WindowsMlExecutionError::new(
            "windows_ml_target_unsupported",
            InitializationStage::RuntimeLoad,
            format!(
                "Windows ML runner does not support {}/{}",
                platform.os, platform.arch
            ),
        )),
    }
}

fn resolved_from_machine_report(
    selected: &SelectedNativeAdapter,
    request: &BackendInitRequest,
    initialization_ms: u64,
    report: &WindowsMlMachineReport,
) -> ResolvedBackend {
    let mut resolved = selected.resolved_backend(request, initialization_ms);
    resolved.configured_provider = report
        .provider_name()
        .map(|provider| format!("Windows ML: {provider}"));
    resolved.accelerator = report.accelerator_name().map(ToOwned::to_owned);
    resolved.execution_plan = ExecutionPlan::Unknown;
    resolved.runtime_version = report.runtime_version();
    resolved
}

fn runner_stage(stage: &str) -> InitializationStage {
    match stage {
        "model-identity" => InitializationStage::ArtifactIntegrity,
        "input-validation" => InitializationStage::BufferPrepare,
        "device-selection" => InitializationStage::DeviceCreate,
        "session-creation" => InitializationStage::ModelCompile,
        "metadata-validation" => InitializationStage::IoDiscovery,
        "inference" | "output-validation" | "artifact-publication" => {
            InitializationStage::SmokeInference
        }
        "catalog-registration"
        | "platform-validation"
        | "provider-introspection"
        | "module-identity"
        | "dependency-identity"
        | "sdk-runtime-identity"
        | "argument-validation" => InitializationStage::RuntimeLoad,
        _ => InitializationStage::RuntimeLoad,
    }
}

fn require_package(
    package: Option<&WindowsMlPackageIdentity>,
    expected_id: &str,
    label: &str,
) -> Result<(), String> {
    let package = package.ok_or_else(|| format!("runner omitted {label}"))?;
    if package.id != expected_id || package.version != WINDOWS_ML_PACKAGE_VERSION {
        return Err(format!(
            "runner {label} is {}/{}, expected {expected_id}/{WINDOWS_ML_PACKAGE_VERSION}",
            package.id, package.version
        ));
    }
    Ok(())
}

fn failure(
    request: &BackendInitRequest,
    code: impl Into<String>,
    stage: InitializationStage,
    message: impl Into<String>,
) -> InitFailure {
    InitFailure::new(code, stage, message).with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::WindowsMl,
    )
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn unavailable_machine_report() -> WindowsMlMachineReport {
    WindowsMlMachineReport {
        schema_version: WINDOWS_ML_RUNNER_SCHEMA_VERSION,
        state: "unavailable".to_owned(),
        target: None,
        runtime_executed: false,
        failure_stage: None,
        windows_ml_api_called: false,
        catalog_registration_attempted: false,
        catalog_registration_completed: false,
        session_created: false,
        inference_executed: false,
        runtime_introspection_complete: false,
        output_published: false,
        runtime: None,
        execution: None,
        error: None,
    }
}
