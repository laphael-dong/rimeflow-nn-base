//! HarmonyOS MindSpore Lite adapter contract and manifest-driven tensor mapping.
//!
//! The official MindSpore Lite runtime is reached through [`MindSporeLiteRuntime`].
//! Keeping that NDK boundary trait-based lets Linux host tests validate artifact,
//! role, lifecycle, and error semantics without claiming a HarmonyOS execution.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{
    BackendInitRequest, BackendInstance, BackendKind, CapabilityStatus, DType, ExecutionPlan,
    ModelInput, NativeAdapterCapability, Platform, RawModelOutput, RawTensor, ResolvedBackend,
    RuntimeBackend, SelectedNativeAdapter, TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::{Artifact, ArtifactFormat, Layout, ModelManifest, TensorSpec};

pub const MINDSPORE_LITE_RUNTIME_VERSION: &str = "2.7.0";
pub const MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS: u64 = 30_000;
pub const MINDSPORE_LITE_INFERENCE_TIMEOUT_MS: u64 = 30_000;

const MINDSPORE_LITE_PROVIDER: &str = "MindSpore Lite NDK";
const MINDSPORE_LITE_TARGET_OS: &str = "harmonyos";
const MINDSPORE_LITE_TARGET_ARCH: &str = "arm64-v8a";

/// Tensor metadata discovered from the official MindSpore Lite NDK session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteTensorDescriptor {
    pub name: String,
    pub index: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteTensorBinding {
    pub role: String,
    pub runtime_name: String,
    pub runtime_index: usize,
    pub shape: Vec<usize>,
    pub layout: Layout,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteIoPlan {
    pub inputs: Vec<MindSporeLiteTensorBinding>,
    pub outputs: Vec<MindSporeLiteTensorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteRuntimeError {
    pub code: String,
    pub message: String,
}

impl MindSporeLiteRuntimeError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for MindSporeLiteRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for MindSporeLiteRuntimeError {}

/// Boundary implemented by the official MindSpore Lite NDK session.
///
/// Production code must populate descriptors from the loaded `.ms` artifact,
/// run its fixed smoke input once, and never use a host or ORT substitute.
pub trait MindSporeLiteRuntime: Send {
    fn input_descriptors(&self) -> &[MindSporeLiteTensorDescriptor];
    fn output_descriptors(&self) -> &[MindSporeLiteTensorDescriptor];
    fn run(
        &mut self,
        inputs: &[TensorData],
        timeout: Duration,
    ) -> Result<Vec<TensorData>, MindSporeLiteRuntimeError>;
}

/// Loads the official target runtime after artifact identity and bytes are verified.
///
/// A HarmonyOS implementation must enforce `timeout` at the NDK boundary and
/// return the measured load duration. Host implementations are test doubles only.
pub trait MindSporeLiteRuntimeLoader: Send + Sync {
    type Runtime: MindSporeLiteRuntime;

    fn load(
        &self,
        artifact: &VerifiedMindSporeLiteArtifact,
        artifact_bytes: &[u8],
        timeout: Duration,
    ) -> Result<MindSporeLiteLoadedRuntime<Self::Runtime>, MindSporeLiteBootstrapError>;
}

pub struct MindSporeLiteLoadedRuntime<R> {
    pub runtime: R,
    pub initialization_ms: u64,
}

/// Host-compilable builder for the target-specific MindSpore Lite NDK loader.
pub struct MindSporeLiteAdapterBuilder<L> {
    manifest: ModelManifest,
    artifact_bytes: Arc<[u8]>,
    loader: L,
}

impl<L: MindSporeLiteRuntimeLoader> MindSporeLiteAdapterBuilder<L> {
    pub fn new(manifest: ModelManifest, artifact_bytes: impl Into<Arc<[u8]>>, loader: L) -> Self {
        Self {
            manifest,
            artifact_bytes: artifact_bytes.into(),
            loader,
        }
    }

    pub fn build(
        &self,
        request: &BackendInitRequest,
        selected: &SelectedNativeAdapter,
    ) -> Result<BackendInstance<MindSporeLiteBackend<L::Runtime>>, InitFailure> {
        validate_selected(request, selected)?;
        let verified = VerifiedMindSporeLiteArtifact::verify(
            self.manifest.clone(),
            request,
            &self.artifact_bytes,
        )?;
        let timeout = Duration::from_millis(MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS);
        let loaded = self
            .loader
            .load(&verified, &self.artifact_bytes, timeout)
            .map_err(|error| error.into_init_failure(request))?;
        if loaded.initialization_ms > MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS {
            return Err(init_failure(
                request,
                "native_initialization_timeout",
                InitializationStage::RuntimeLoad,
                format!(
                    "MindSpore Lite runtime load exceeded {MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS} ms"
                ),
            ));
        }

        let resolved = selected.resolved_backend(request, loaded.initialization_ms);
        let smoke_timeout = Duration::from_millis(
            MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS - loaded.initialization_ms,
        );
        let backend = MindSporeLiteBackend::from_verified_artifact(
            verified,
            loaded.runtime,
            resolved.clone(),
            smoke_timeout,
        )?;
        Ok(BackendInstance { backend, resolved })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindSporeLiteBootstrapError {
    pub code: &'static str,
    pub stage: InitializationStage,
    pub message: String,
}

impl MindSporeLiteBootstrapError {
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(
            "native_runtime_unavailable",
            InitializationStage::RuntimeLoad,
            message,
        )
    }

    pub fn device(message: impl Into<String>) -> Self {
        Self::new(
            "native_device_unavailable",
            InitializationStage::DeviceCreate,
            message,
        )
    }

    pub fn model_load(message: impl Into<String>) -> Self {
        Self::new(
            "mindspore_lite_model_load_failed",
            InitializationStage::ModelCompile,
            message,
        )
    }

    pub fn io_discovery(message: impl Into<String>) -> Self {
        Self::new(
            "mindspore_lite_io_contract_mismatch",
            InitializationStage::IoDiscovery,
            message,
        )
    }

    pub fn buffer_prepare(message: impl Into<String>) -> Self {
        Self::new(
            "mindspore_lite_buffer_prepare_failed",
            InitializationStage::BufferPrepare,
            message,
        )
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self::new(
            "native_initialization_timeout",
            InitializationStage::RuntimeLoad,
            message,
        )
    }

    fn new(code: &'static str, stage: InitializationStage, message: impl Into<String>) -> Self {
        Self {
            code,
            stage,
            message: message.into(),
        }
    }

    pub fn into_init_failure(self, request: &BackendInitRequest) -> InitFailure {
        InitFailure::new(self.code, self.stage, self.message).with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::MindSporeLite,
        )
    }
}

/// Token proving that the requested artifact's manifest identity and bytes
/// were checked before the NDK runtime was allowed to load it.
#[derive(Debug, Clone)]
pub struct VerifiedMindSporeLiteArtifact {
    manifest: ModelManifest,
    artifact: Artifact,
    request: BackendInitRequest,
}

impl VerifiedMindSporeLiteArtifact {
    pub fn verify(
        manifest: ModelManifest,
        request: &BackendInitRequest,
        artifact_bytes: &[u8],
    ) -> Result<Self, InitFailure> {
        manifest.validate_semantics().map_err(|error| {
            init_failure(
                request,
                "manifest_invalid",
                InitializationStage::ManifestParse,
                error.to_string(),
            )
        })?;
        if request.target.os != MINDSPORE_LITE_TARGET_OS
            || request.target.arch != MINDSPORE_LITE_TARGET_ARCH
        {
            return Err(init_failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                format!(
                    "MindSpore Lite adapter requires {MINDSPORE_LITE_TARGET_OS}/{MINDSPORE_LITE_TARGET_ARCH}"
                ),
            ));
        }
        if manifest.model.id != request.model_id || manifest.model.version != request.model_version
        {
            return Err(init_failure(
                request,
                "manifest_identity_mismatch",
                InitializationStage::ManifestParse,
                "manifest model identity does not match the initialization request",
            ));
        }
        let artifact = manifest
            .select_artifact(&request.artifact_id, &request.target)
            .map_err(|error| {
                init_failure(
                    request,
                    "adapter_or_artifact_unavailable",
                    InitializationStage::ArtifactIntegrity,
                    error.to_string(),
                )
            })?
            .clone();
        if artifact.format != ArtifactFormat::MindsporeLite
            || !artifact.path.ends_with(".ms")
            || artifact.sha256 != request.artifact_sha256
        {
            return Err(init_failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                "MindSpore Lite requires the request's manifest-selected .ms artifact",
            ));
        }
        ModelManifest::verify_artifact_bytes(&artifact, artifact_bytes).map_err(|error| {
            init_failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                error.to_string(),
            )
        })?;
        Ok(Self {
            manifest,
            artifact,
            request: request.clone(),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    pub fn request(&self) -> &BackendInitRequest {
        &self.request
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteDiagnostics {
    pub resolved: ResolvedBackend,
    pub io_plan: MindSporeLiteIoPlan,
    pub smoke_inference_completed: bool,
    pub runtime_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MindSporeLiteAvailability {
    pub artifact: CapabilityStatus,
    pub runtime: CapabilityStatus,
    pub device: CapabilityStatus,
    pub smoke: CapabilityStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<String>,
}

impl MindSporeLiteAvailability {
    pub fn into_native_capability(self) -> NativeAdapterCapability {
        NativeAdapterCapability {
            backend_kind: BackendKind::MindSporeLite,
            target: Platform::new(MINDSPORE_LITE_TARGET_OS, MINDSPORE_LITE_TARGET_ARCH),
            artifact_formats: vec![ArtifactFormat::MindsporeLite],
            artifact: self.artifact,
            runtime: self.runtime,
            device: self.device,
            smoke: self.smoke,
            configured_provider: Some(MINDSPORE_LITE_PROVIDER.to_owned()),
            accelerator: self.accelerator,
            execution_plan: ExecutionPlan::Unknown,
            runtime_version: Some(MINDSPORE_LITE_RUNTIME_VERSION.to_owned()),
        }
    }
}

pub struct MindSporeLiteBackend<R> {
    runtime: R,
    diagnostics: MindSporeLiteDiagnostics,
}

impl<R: MindSporeLiteRuntime> MindSporeLiteBackend<R> {
    fn from_verified_artifact(
        verified: VerifiedMindSporeLiteArtifact,
        mut runtime: R,
        resolved: ResolvedBackend,
        smoke_timeout: Duration,
    ) -> Result<Self, InitFailure> {
        validate_resolved(&verified, &resolved)?;
        let io_plan = MindSporeLiteIoPlan::from_verified(
            &verified,
            runtime.input_descriptors(),
            runtime.output_descriptors(),
        )?;
        let smoke_inputs = io_plan
            .inputs
            .iter()
            .map(smoke_tensor)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|message| smoke_failure(verified.request(), message))?;
        let smoke_outputs = runtime
            .run(&smoke_inputs, smoke_timeout)
            .map_err(|error| runtime_smoke_failure(verified.request(), error))?;
        validate_outputs(&io_plan.outputs, smoke_outputs)
            .map_err(|error| smoke_failure(verified.request(), error.message))?;

        Ok(Self {
            runtime,
            diagnostics: MindSporeLiteDiagnostics {
                resolved,
                io_plan,
                smoke_inference_completed: true,
                runtime_version: MINDSPORE_LITE_RUNTIME_VERSION.to_owned(),
            },
        })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.diagnostics.resolved
    }

    pub fn diagnostics(&self) -> &MindSporeLiteDiagnostics {
        &self.diagnostics
    }
}

impl<R: MindSporeLiteRuntime> RuntimeBackend for MindSporeLiteBackend<R> {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        input.validate()?;
        let ModelInput::Tensor {
            role,
            shape,
            dtype,
            bytes,
        } = input
        else {
            return Err(InferenceError::new(
                "unsupported_input",
                "MindSpore Lite requires a preprocessed manifest-layout tensor",
            ));
        };
        let Some(binding) = self.diagnostics.io_plan.inputs.first() else {
            return Err(InferenceError::new(
                "mindspore_lite_io_contract_mismatch",
                "MindSpore Lite input plan is empty",
            ));
        };
        if self.diagnostics.io_plan.inputs.len() != 1 {
            return Err(InferenceError::new(
                "unsupported_input_arity",
                "ModelInput can supply exactly one logical MindSpore Lite input",
            ));
        }
        if binding.role != role {
            return Err(InferenceError::new(
                "input_role_mismatch",
                format!("expected logical role {}, got {role}", binding.role),
            ));
        }
        if binding.shape != shape {
            return Err(InferenceError::new(
                "input_shape_mismatch",
                format!("expected {:?}, got {shape:?}", binding.shape),
            ));
        }
        if dtype != binding.dtype {
            return Err(InferenceError::new(
                "input_dtype_mismatch",
                format!("expected {:?}, got {dtype:?}", binding.dtype),
            ));
        }
        let tensor = tensor_data(binding.dtype, &bytes)?;
        let outputs = self
            .runtime
            .run(
                &[tensor],
                Duration::from_millis(MINDSPORE_LITE_INFERENCE_TIMEOUT_MS),
            )
            .map_err(|error| InferenceError::new(error.code, error.message))?;
        validate_outputs(&self.diagnostics.io_plan.outputs, outputs)
            .map_err(|error| InferenceError::new(error.code, error.message))
    }
}

impl MindSporeLiteIoPlan {
    fn from_verified(
        verified: &VerifiedMindSporeLiteArtifact,
        runtime_inputs: &[MindSporeLiteTensorDescriptor],
        runtime_outputs: &[MindSporeLiteTensorDescriptor],
    ) -> Result<Self, InitFailure> {
        if verified.artifact.inputs.len() != 1 || verified.artifact.outputs.len() != 1 {
            return Err(io_failure(
                verified.request(),
                "the first MindSpore Lite adapter requires exactly one input and one output role",
            ));
        }
        let inputs = map_bindings(
            &verified.artifact.inputs,
            &verified.manifest.tensors.inputs,
            runtime_inputs,
            "input",
            verified.request(),
        )?;
        let outputs = map_bindings(
            &verified.artifact.outputs,
            &verified.manifest.tensors.outputs,
            runtime_outputs,
            "output",
            verified.request(),
        )?;
        let input = &inputs[0];
        if input.role != "image"
            || input.runtime_name != "images"
            || input.runtime_index != 0
            || input.layout != Layout::Nhwc
            || input.dtype != DType::F32
            || input.shape != [1, 640, 640, 3]
        {
            return Err(io_failure(
                verified.request(),
                "MindSpore Lite requires image/images/index-0 FP32 NHWC [1,640,640,3]",
            ));
        }
        let output = &outputs[0];
        if output.role != "detections"
            || output.runtime_name != "output0"
            || output.runtime_index != 0
            || output.dtype != DType::F32
            || output.shape != [1, 84, 8400]
        {
            return Err(io_failure(
                verified.request(),
                "MindSpore Lite requires detections/output0/index-0 FP32 [1,84,8400]",
            ));
        }
        Ok(Self { inputs, outputs })
    }
}

fn map_bindings(
    roles: &[String],
    specs: &[TensorSpec],
    runtime: &[MindSporeLiteTensorDescriptor],
    label: &str,
    request: &BackendInitRequest,
) -> Result<Vec<MindSporeLiteTensorBinding>, InitFailure> {
    if roles.len() != runtime.len() {
        return Err(io_failure(
            request,
            format!("MindSpore Lite {label} descriptor count differs from the artifact role count"),
        ));
    }
    roles
        .iter()
        .map(|role| {
            let spec = specs
                .iter()
                .find(|spec| spec.role == *role)
                .ok_or_else(|| {
                    io_failure(
                        request,
                        format!("MindSpore Lite {label} role {role} is absent"),
                    )
                })?;
            if spec.quantization.is_some() {
                return Err(io_failure(
                    request,
                    format!("MindSpore Lite FP32 {label} role {role} must not be quantized"),
                ));
            }
            let expected_shape = shape_to_usize(spec, request)?;
            let descriptor = runtime
                .iter()
                .find(|candidate| {
                    spec.name.as_deref() == Some(candidate.name.as_str())
                        && spec.index == Some(candidate.index)
                })
                .ok_or_else(|| {
                    io_failure(
                        request,
                        format!(
                            "MindSpore Lite {label} role {role} was not discovered by name/index"
                        ),
                    )
                })?;
            if descriptor.shape != expected_shape || descriptor.dtype != spec.dtype {
                return Err(io_failure(
                    request,
                    format!("MindSpore Lite {label} role {role} descriptor differs from manifest"),
                ));
            }
            Ok(MindSporeLiteTensorBinding {
                role: spec.role.clone(),
                runtime_name: descriptor.name.clone(),
                runtime_index: descriptor.index,
                shape: expected_shape,
                layout: spec.layout,
                dtype: spec.dtype,
            })
        })
        .collect()
}

fn shape_to_usize(
    spec: &TensorSpec,
    request: &BackendInitRequest,
) -> Result<Vec<usize>, InitFailure> {
    spec.shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                io_failure(
                    request,
                    format!("MindSpore Lite tensor {} has a non-static shape", spec.role),
                )
            })
        })
        .collect()
}

fn smoke_tensor(binding: &MindSporeLiteTensorBinding) -> Result<TensorData, String> {
    let count = binding
        .shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| format!("smoke input {} shape overflows", binding.role))?;
    match binding.dtype {
        DType::F32 => Ok(TensorData::F32(vec![0.0; count])),
        _ => Err(format!(
            "MindSpore Lite frozen smoke contract requires FLOAT32, got {:?}",
            binding.dtype
        )),
    }
}

fn tensor_data(dtype: DType, bytes: &[u8]) -> Result<TensorData, InferenceError> {
    match dtype {
        DType::F32 => Ok(TensorData::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks have four bytes")))
                .collect(),
        )),
        _ => Err(InferenceError::new(
            "mindspore_lite_dtype_unsupported",
            "the frozen MindSpore Lite artifact accepts only FP32 input",
        )),
    }
}

#[derive(Debug)]
struct OutputError {
    code: &'static str,
    message: String,
}

fn validate_outputs(
    bindings: &[MindSporeLiteTensorBinding],
    outputs: Vec<TensorData>,
) -> Result<RawModelOutput, OutputError> {
    if bindings.len() != outputs.len() {
        return Err(OutputError {
            code: "mindspore_lite_output_arity_mismatch",
            message: format!("expected {} outputs, got {}", bindings.len(), outputs.len()),
        });
    }
    let mut tensors = Vec::with_capacity(bindings.len());
    for (binding, data) in bindings.iter().zip(outputs) {
        if data_dtype(&data) != binding.dtype {
            return Err(OutputError {
                code: "mindspore_lite_output_dtype_mismatch",
                message: format!("output {} has an unexpected dtype", binding.role),
            });
        }
        let expected = binding.shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(OutputError {
                code: "mindspore_lite_output_shape_mismatch",
                message: format!(
                    "output {} expected {expected} elements, got {}",
                    binding.role,
                    data.len()
                ),
            });
        }
        if let TensorData::F32(values) = &data {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(OutputError {
                    code: "mindspore_lite_output_non_finite",
                    message: format!("output {} contains non-finite values", binding.role),
                });
            }
        }
        tensors.push(RawTensor {
            role: binding.role.clone(),
            shape: binding.shape.clone(),
            data,
        });
    }
    Ok(RawModelOutput { tensors })
}

fn data_dtype(data: &TensorData) -> DType {
    match data {
        TensorData::F32(_) => DType::F32,
        TensorData::F16(_) => DType::F16,
        TensorData::I8(_) => DType::I8,
        TensorData::U8(_) => DType::U8,
        TensorData::I32(_) => DType::I32,
        TensorData::I64(_) => DType::I64,
        TensorData::Bool(_) => DType::Bool,
    }
}

fn validate_resolved(
    verified: &VerifiedMindSporeLiteArtifact,
    resolved: &ResolvedBackend,
) -> Result<(), InitFailure> {
    let request = verified.request();
    if resolved.backend_kind != BackendKind::MindSporeLite
        || resolved.platform != request.target
        || resolved.model_version != request.model_version
        || resolved.artifact_id != request.artifact_id
        || resolved.artifact_sha256 != request.artifact_sha256
        || resolved.configured_provider.as_deref() != Some(MINDSPORE_LITE_PROVIDER)
        || resolved.runtime_version.as_deref() != Some(MINDSPORE_LITE_RUNTIME_VERSION)
    {
        return Err(init_failure(
            request,
            "native_factory_diagnostic_mismatch",
            InitializationStage::SmokeInference,
            "MindSpore Lite diagnostics do not match the verified artifact request",
        ));
    }
    Ok(())
}

fn validate_selected(
    request: &BackendInitRequest,
    selected: &SelectedNativeAdapter,
) -> Result<(), InitFailure> {
    if selected.backend_kind != BackendKind::MindSporeLite
        || selected.platform != request.target
        || selected.artifact_id != request.artifact_id
        || selected.artifact_sha256 != request.artifact_sha256
        || selected.configured_provider.as_deref() != Some(MINDSPORE_LITE_PROVIDER)
        || selected.runtime_version.as_deref() != Some(MINDSPORE_LITE_RUNTIME_VERSION)
    {
        return Err(init_failure(
            request,
            "native_factory_diagnostic_mismatch",
            InitializationStage::RuntimeLoad,
            "MindSpore Lite selection does not match the requested artifact and runtime",
        ));
    }
    Ok(())
}

fn init_failure(
    request: &BackendInitRequest,
    code: &'static str,
    stage: InitializationStage,
    message: impl Into<String>,
) -> InitFailure {
    InitFailure::new(code, stage, message).with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::MindSporeLite,
    )
}

fn io_failure(request: &BackendInitRequest, message: impl Into<String>) -> InitFailure {
    init_failure(
        request,
        "mindspore_lite_io_contract_mismatch",
        InitializationStage::IoDiscovery,
        message,
    )
}

fn smoke_failure(request: &BackendInitRequest, message: impl Into<String>) -> InitFailure {
    init_failure(
        request,
        "native_smoke_failed",
        InitializationStage::SmokeInference,
        message,
    )
}

fn runtime_smoke_failure(
    request: &BackendInitRequest,
    error: MindSporeLiteRuntimeError,
) -> InitFailure {
    let code = if error.code == "native_initialization_timeout" {
        "native_initialization_timeout"
    } else {
        "native_smoke_failed"
    };
    init_failure(
        request,
        code,
        InitializationStage::SmokeInference,
        error.to_string(),
    )
}
