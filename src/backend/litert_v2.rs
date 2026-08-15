//! Android LiteRT v2 adapter contract and manifest-driven tensor mapping.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::{
    BackendInitRequest, BackendKind, CapabilityStatus, DType, ExecutionPlan, ModelInput,
    NativeAdapterCapability, Platform, RawModelOutput, RawTensor, ResolvedBackend, RuntimeBackend,
    TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::{Artifact, ArtifactFormat, Layout, ModelManifest, TensorSpec};

pub const LITERT_RUNTIME_VERSION: &str = "2.1.6";
pub const LITERT_RUST_BINDING_VERSION: &str = "0.1.3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiteRtTensorDescriptor {
    pub name: String,
    pub index: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiteRtTensorBinding {
    pub role: String,
    pub runtime_name: String,
    pub runtime_index: usize,
    pub shape: Vec<usize>,
    pub layout: Layout,
    pub dtype: DType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_zero_point: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiteRtIoPlan {
    pub inputs: Vec<LiteRtTensorBinding>,
    pub outputs: Vec<LiteRtTensorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiteRtRuntimeError {
    pub code: String,
    pub message: String,
}

impl LiteRtRuntimeError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for LiteRtRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for LiteRtRuntimeError {}

/// Boundary implemented by the official LiteRT `CompiledModel` runtime.
/// Test doubles may implement it for contract tests, but production inference
/// is provided only by the Android implementation below.
pub trait LiteRtCompiledRuntime: Send {
    fn input_descriptors(&self) -> &[LiteRtTensorDescriptor];
    fn output_descriptors(&self) -> &[LiteRtTensorDescriptor];
    fn run(&mut self, inputs: &[TensorData]) -> Result<Vec<TensorData>, LiteRtRuntimeError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteRtV2BootstrapError {
    pub code: &'static str,
    pub stage: InitializationStage,
    pub message: String,
}

impl LiteRtV2BootstrapError {
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

    pub fn model_compile(message: impl Into<String>) -> Self {
        Self::new(
            "litert_model_compile_failed",
            InitializationStage::ModelCompile,
            message,
        )
    }

    pub fn io_discovery(message: impl Into<String>) -> Self {
        Self::new(
            "litert_io_contract_mismatch",
            InitializationStage::IoDiscovery,
            message,
        )
    }

    pub fn buffer_prepare(message: impl Into<String>) -> Self {
        Self::new(
            "litert_buffer_prepare_failed",
            InitializationStage::BufferPrepare,
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
            BackendKind::LiteRtV2,
        )
    }
}

/// Manifest and digest verification token. Its private fields prevent a
/// compiled runtime from being published before artifact verification.
#[derive(Debug, Clone)]
pub struct VerifiedLiteRtArtifact {
    manifest: ModelManifest,
    artifact: Artifact,
    request: BackendInitRequest,
}

impl VerifiedLiteRtArtifact {
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
        if artifact.format != ArtifactFormat::Tflite || artifact.sha256 != request.artifact_sha256 {
            return Err(init_failure(
                request,
                "adapter_or_artifact_unavailable",
                InitializationStage::ArtifactIntegrity,
                "LiteRT requires the request's manifest-selected TFLite artifact",
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
pub struct LiteRtV2Diagnostics {
    pub resolved: ResolvedBackend,
    pub io_plan: LiteRtIoPlan,
    pub smoke_inference_completed: bool,
    pub runtime_version: String,
    pub rust_binding_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LiteRtV2Availability {
    pub artifact: CapabilityStatus,
    pub runtime: CapabilityStatus,
    pub device: CapabilityStatus,
    pub smoke: CapabilityStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accelerator: Option<String>,
}

impl LiteRtV2Availability {
    pub fn into_native_capability(self, target: Platform) -> NativeAdapterCapability {
        NativeAdapterCapability {
            backend_kind: BackendKind::LiteRtV2,
            target,
            artifact_formats: vec![ArtifactFormat::Tflite],
            artifact: self.artifact,
            runtime: self.runtime,
            device: self.device,
            smoke: self.smoke,
            configured_provider: Some("LiteRT CompiledModel".to_owned()),
            accelerator: self.accelerator,
            execution_plan: ExecutionPlan::Unknown,
            runtime_version: Some(LITERT_RUNTIME_VERSION.to_owned()),
        }
    }
}

pub struct LiteRtV2Backend<R> {
    runtime: R,
    diagnostics: LiteRtV2Diagnostics,
}

impl<R: LiteRtCompiledRuntime> LiteRtV2Backend<R> {
    pub fn from_verified_artifact(
        verified: VerifiedLiteRtArtifact,
        mut runtime: R,
        resolved: ResolvedBackend,
    ) -> Result<Self, InitFailure> {
        validate_resolved(&verified, &resolved)?;
        let io_plan = LiteRtIoPlan::from_verified(
            &verified,
            runtime.input_descriptors(),
            runtime.output_descriptors(),
        )?;
        let smoke_inputs = io_plan
            .inputs
            .iter()
            .map(smoke_tensor)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|message| smoke_failure(&verified.request, message))?;
        let smoke_outputs = runtime
            .run(&smoke_inputs)
            .map_err(|error| smoke_failure(&verified.request, error.to_string()))?;
        validate_outputs(&io_plan.outputs, smoke_outputs)
            .map_err(|error| smoke_failure(&verified.request, error.message))?;

        Ok(Self {
            runtime,
            diagnostics: LiteRtV2Diagnostics {
                resolved,
                io_plan,
                smoke_inference_completed: true,
                runtime_version: LITERT_RUNTIME_VERSION.to_owned(),
                rust_binding_version: LITERT_RUST_BINDING_VERSION.to_owned(),
            },
        })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.diagnostics.resolved
    }

    pub fn diagnostics(&self) -> &LiteRtV2Diagnostics {
        &self.diagnostics
    }

    #[cfg(all(target_os = "android", feature = "litert-v2"))]
    pub(crate) fn set_initialization_ms(&mut self, initialization_ms: u64) {
        self.diagnostics.resolved.initialization_ms = initialization_ms;
    }
}

impl<R: LiteRtCompiledRuntime> RuntimeBackend for LiteRtV2Backend<R> {
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
                "LiteRT v2 requires a preprocessed manifest-layout tensor",
            ));
        };
        let Some(binding) = self.diagnostics.io_plan.inputs.first() else {
            return Err(InferenceError::new(
                "litert_io_contract_mismatch",
                "LiteRT input plan is empty",
            ));
        };
        if self.diagnostics.io_plan.inputs.len() != 1 {
            return Err(InferenceError::new(
                "unsupported_input_arity",
                "ModelInput can supply exactly one logical LiteRT input",
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
        let tensor = input_tensor_data(binding, dtype, &bytes)?;
        let outputs = self
            .runtime
            .run(std::slice::from_ref(&tensor))
            .map_err(|error| InferenceError::new("litert_inference_failed", error.to_string()))?;
        validate_outputs(&self.diagnostics.io_plan.outputs, outputs)
    }
}

impl LiteRtIoPlan {
    fn from_verified(
        verified: &VerifiedLiteRtArtifact,
        runtime_inputs: &[LiteRtTensorDescriptor],
        runtime_outputs: &[LiteRtTensorDescriptor],
    ) -> Result<Self, InitFailure> {
        let inputs = map_bindings(
            &verified.artifact.inputs,
            &verified.manifest.tensors.inputs,
            runtime_inputs,
            "input",
        )
        .map_err(|message| io_failure(&verified.request, message))?;
        let outputs = map_bindings(
            &verified.artifact.outputs,
            &verified.manifest.tensors.outputs,
            runtime_outputs,
            "output",
        )
        .map_err(|message| io_failure(&verified.request, message))?;
        if inputs.len() != 1 {
            return Err(io_failure(
                &verified.request,
                "LiteRT adapter currently requires exactly one logical input",
            ));
        }
        Ok(Self { inputs, outputs })
    }
}

fn map_bindings(
    roles: &[String],
    specs: &[TensorSpec],
    runtime: &[LiteRtTensorDescriptor],
    kind: &str,
) -> Result<Vec<LiteRtTensorBinding>, String> {
    if roles.len() != runtime.len() {
        return Err(format!(
            "manifest declares {} {kind}(s), runtime exposes {}",
            roles.len(),
            runtime.len()
        ));
    }
    let mut runtime_indices = HashSet::new();
    let mut runtime_names = HashSet::new();
    for descriptor in runtime {
        if !runtime_indices.insert(descriptor.index)
            || (!descriptor.name.is_empty() && !runtime_names.insert(descriptor.name.as_str()))
        {
            return Err(format!("runtime exposes duplicate {kind} identity"));
        }
    }

    let mut bindings = roles
        .iter()
        .map(|role| {
            let spec = specs
                .iter()
                .find(|candidate| candidate.role == *role)
                .ok_or_else(|| format!("unknown logical {kind} role {role}"))?;
            if !matches!(spec.dtype, DType::F32 | DType::U8 | DType::I8) {
                return Err(format!(
                    "LiteRT {kind} role {role} has unsupported dtype {:?}",
                    spec.dtype
                ));
            }
            let mut candidates = runtime.iter().filter(|descriptor| {
                spec.name
                    .as_ref()
                    .is_none_or(|name| descriptor.name == *name)
                    && spec.index.is_none_or(|index| descriptor.index == index)
            });
            let descriptor = candidates
                .next()
                .ok_or_else(|| format!("no runtime {kind} matches logical role {role}"))?;
            if candidates.next().is_some() {
                return Err(format!(
                    "runtime {kind} mapping for role {role} is ambiguous"
                ));
            }
            let shape = spec
                .shape
                .iter()
                .map(|dimension| {
                    usize::try_from(*dimension)
                        .map_err(|_| format!("invalid static shape for role {role}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if descriptor.shape != shape || descriptor.dtype != spec.dtype {
                return Err(format!(
                    "runtime {kind} for role {role} is {:?}/{:?}, manifest requires {:?}/{:?}",
                    descriptor.shape, descriptor.dtype, shape, spec.dtype
                ));
            }
            let (quantization_scale, quantization_zero_point) = spec
                .quantization
                .as_ref()
                .map_or((None, None), |quantization| {
                    (Some(quantization.scale), quantization.zero_point)
                });
            Ok(LiteRtTensorBinding {
                role: role.clone(),
                runtime_name: descriptor.name.clone(),
                runtime_index: descriptor.index,
                shape,
                layout: spec.layout,
                dtype: spec.dtype,
                quantization_scale,
                quantization_zero_point,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    bindings.sort_by_key(|binding| binding.runtime_index);
    Ok(bindings)
}

fn input_tensor_data(
    binding: &LiteRtTensorBinding,
    source_dtype: DType,
    bytes: &[u8],
) -> Result<TensorData, InferenceError> {
    match (binding.dtype, source_dtype) {
        (DType::F32, DType::F32) => decode_f32(bytes).map(TensorData::F32),
        (DType::U8, DType::U8) => Ok(TensorData::U8(bytes.to_vec())),
        (DType::I8, DType::I8) => Ok(TensorData::I8(
            bytes.iter().map(|value| *value as i8).collect(),
        )),
        (DType::U8 | DType::I8, DType::F32) => {
            let values = decode_f32(bytes)?;
            quantize_f32(binding, &values)
        }
        _ => Err(InferenceError::new(
            "input_dtype_mismatch",
            format!(
                "role {} expects {:?}, got {:?}",
                binding.role, binding.dtype, source_dtype
            ),
        )),
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, InferenceError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(InferenceError::new(
            "invalid_tensor_bytes",
            "f32 tensor byte count is not divisible by four",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

pub fn quantize_f32(
    binding: &LiteRtTensorBinding,
    values: &[f32],
) -> Result<TensorData, InferenceError> {
    let scale = binding.quantization_scale.ok_or_else(|| {
        InferenceError::new(
            "quantization_parameters_missing",
            format!("role {} has no quantization scale", binding.role),
        )
    })?;
    let zero_point = binding.quantization_zero_point.ok_or_else(|| {
        InferenceError::new(
            "quantization_parameters_missing",
            format!("role {} has no quantization zero point", binding.role),
        )
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InferenceError::new(
            "non_finite_input",
            "quantized LiteRT input contains a non-finite value",
        ));
    }
    match binding.dtype {
        DType::U8 => Ok(TensorData::U8(
            values
                .iter()
                .map(|value| {
                    ((*value as f64 / scale).round() + zero_point as f64).clamp(0.0, 255.0) as u8
                })
                .collect(),
        )),
        DType::I8 => Ok(TensorData::I8(
            values
                .iter()
                .map(|value| {
                    ((*value as f64 / scale).round() + zero_point as f64).clamp(-128.0, 127.0) as i8
                })
                .collect(),
        )),
        _ => Err(InferenceError::new(
            "quantization_dtype_invalid",
            format!("role {} is not an i8/u8 tensor", binding.role),
        )),
    }
}

fn smoke_tensor(binding: &LiteRtTensorBinding) -> Result<TensorData, String> {
    let elements = element_count(&binding.shape)
        .ok_or_else(|| format!("smoke input shape for role {} overflows", binding.role))?;
    match binding.dtype {
        DType::F32 => Ok(TensorData::F32(vec![0.0; elements])),
        DType::U8 => {
            let zero_point = binding
                .quantization_zero_point
                .ok_or_else(|| format!("role {} has no u8 zero point", binding.role))?;
            Ok(TensorData::U8(vec![zero_point as u8; elements]))
        }
        DType::I8 => {
            let zero_point = binding
                .quantization_zero_point
                .ok_or_else(|| format!("role {} has no i8 zero point", binding.role))?;
            Ok(TensorData::I8(vec![zero_point as i8; elements]))
        }
        _ => Err(format!(
            "role {} has unsupported LiteRT dtype {:?}",
            binding.role, binding.dtype
        )),
    }
}

fn validate_outputs(
    bindings: &[LiteRtTensorBinding],
    outputs: Vec<TensorData>,
) -> Result<RawModelOutput, InferenceError> {
    if outputs.len() != bindings.len() {
        return Err(InferenceError::new(
            "litert_output_count_mismatch",
            format!("expected {} outputs, got {}", bindings.len(), outputs.len()),
        ));
    }
    let tensors = bindings
        .iter()
        .zip(outputs)
        .map(|(binding, data)| {
            let expected = element_count(&binding.shape).ok_or_else(|| {
                InferenceError::new("output_shape_overflow", "LiteRT output shape overflows")
            })?;
            if tensor_data_dtype(&data) != binding.dtype || data.len() != expected {
                return Err(InferenceError::new(
                    "litert_output_contract_mismatch",
                    format!(
                        "role {} expected {:?}/{} elements, got {:?}/{}",
                        binding.role,
                        binding.dtype,
                        expected,
                        tensor_data_dtype(&data),
                        data.len()
                    ),
                ));
            }
            if matches!(&data, TensorData::F32(values) if values.iter().any(|value| !value.is_finite()))
            {
                return Err(InferenceError::new(
                    "smoke_output_invalid",
                    format!("role {} contains non-finite output", binding.role),
                ));
            }
            Ok(RawTensor {
                role: binding.role.clone(),
                shape: binding.shape.clone(),
                data,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RawModelOutput { tensors })
}

fn tensor_data_dtype(data: &TensorData) -> DType {
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

fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
}

fn validate_resolved(
    verified: &VerifiedLiteRtArtifact,
    resolved: &ResolvedBackend,
) -> Result<(), InitFailure> {
    if resolved.backend_kind != BackendKind::LiteRtV2
        || resolved.platform != verified.request.target
        || resolved.model_version != verified.request.model_version
        || resolved.artifact_id != verified.artifact.id
        || resolved.artifact_sha256 != verified.artifact.sha256
    {
        return Err(init_failure(
            &verified.request,
            "native_factory_diagnostic_mismatch",
            InitializationStage::IoDiscovery,
            "LiteRT diagnostics do not match the verified request and artifact",
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
        BackendKind::LiteRtV2,
    )
}

fn io_failure(request: &BackendInitRequest, message: impl Into<String>) -> InitFailure {
    init_failure(
        request,
        "litert_io_contract_mismatch",
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

#[cfg(all(target_os = "android", feature = "litert-v2"))]
mod android;
#[cfg(all(target_os = "android", feature = "litert-v2"))]
pub use android::{AndroidLiteRtAccelerator, AndroidLiteRtV2Backend, AndroidLiteRtV2Factory};
