//! Direct Linux adapter for the official OpenVINO Runtime C API.

#![cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    feature = "openvino-runtime"
))]

use std::path::PathBuf;
use std::time::Instant;

use openvino::{Core, DeviceType, ElementType, InferRequest, Shape, Tensor};

use crate::backend::{
    BackendKind, DType, ExecutionPlan, ModelInput, Platform, RawModelOutput, RawTensor,
    ResolvedBackend, RuntimeBackend, TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::sha256_hex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenVinoMetadata {
    pub platform: Platform,
    pub model_version: String,
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub input_role: String,
    pub input_shape: Vec<usize>,
    pub output_role: String,
    pub output_shape: Vec<usize>,
}

pub struct OpenVinoBackend {
    request: InferRequest,
    resolved: ResolvedBackend,
    input_role: String,
    input_shape: Vec<usize>,
    output_role: String,
    output_shape: Vec<usize>,
}

impl OpenVinoBackend {
    pub fn from_model_bytes(
        model_bytes: &[u8],
        metadata: OpenVinoMetadata,
    ) -> Result<Self, InitFailure> {
        Self::build(model_bytes, metadata, None)
    }

    pub fn from_model_bytes_with_runtime(
        model_bytes: &[u8],
        runtime_library: impl Into<PathBuf>,
        metadata: OpenVinoMetadata,
    ) -> Result<Self, InitFailure> {
        Self::build(model_bytes, metadata, Some(runtime_library.into()))
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.resolved
    }

    fn build(
        model_bytes: &[u8],
        metadata: OpenVinoMetadata,
        runtime_library: Option<PathBuf>,
    ) -> Result<Self, InitFailure> {
        validate_metadata(model_bytes, &metadata)?;
        let started = Instant::now();

        if let Some(path) = runtime_library.as_deref() {
            openvino_sys::library::load_from(path).map_err(|error| {
                init_failure(
                    &metadata,
                    "openvino_runtime_unavailable",
                    InitializationStage::RuntimeLoad,
                    format!(
                        "failed to load OpenVINO Runtime from {}: {error}",
                        path.display()
                    ),
                )
            })?;
        }

        let mut core = Core::new().map_err(|error| {
            init_failure(
                &metadata,
                "openvino_runtime_unavailable",
                InitializationStage::RuntimeLoad,
                error.to_string(),
            )
        })?;
        let devices = core.available_devices().map_err(|error| {
            init_failure(
                &metadata,
                "openvino_device_discovery_failed",
                InitializationStage::DeviceCreate,
                error.to_string(),
            )
        })?;
        if !devices.contains(&DeviceType::CPU) {
            return Err(init_failure(
                &metadata,
                "openvino_cpu_unavailable",
                InitializationStage::DeviceCreate,
                "OpenVINO Runtime did not expose the CPU device",
            ));
        }
        let runtime_version = core
            .versions("CPU")
            .ok()
            .and_then(|versions| versions.into_iter().next())
            .map(|(_, version)| version.build_number);

        let model = core
            .read_model_from_buffer(model_bytes, None)
            .map_err(|error| {
                init_failure(
                    &metadata,
                    "openvino_model_load_failed",
                    InitializationStage::ModelCompile,
                    error.to_string(),
                )
            })?;
        validate_model_io(&model, &metadata)?;
        let mut compiled = core
            .compile_model(&model, DeviceType::CPU)
            .map_err(|error| {
                init_failure(
                    &metadata,
                    "openvino_model_compile_failed",
                    InitializationStage::ModelCompile,
                    error.to_string(),
                )
            })?;
        let mut request = compiled.create_infer_request().map_err(|error| {
            init_failure(
                &metadata,
                "openvino_buffer_prepare_failed",
                InitializationStage::BufferPrepare,
                error.to_string(),
            )
        })?;

        let smoke_elements = element_count(&metadata.input_shape).map_err(|error| {
            init_failure(
                &metadata,
                "openvino_smoke_failed",
                InitializationStage::SmokeInference,
                error.message,
            )
        })?;
        run_inference(&mut request, &metadata, &vec![0.0; smoke_elements]).map_err(|error| {
            init_failure(
                &metadata,
                "openvino_smoke_failed",
                InitializationStage::SmokeInference,
                error.message,
            )
        })?;

        Ok(Self {
            request,
            resolved: ResolvedBackend {
                backend_kind: BackendKind::OpenVino,
                platform: metadata.platform,
                configured_provider: Some("OpenVINO Runtime".to_owned()),
                accelerator: Some("CPU".to_owned()),
                execution_plan: ExecutionPlan::Full,
                model_version: metadata.model_version,
                artifact_id: metadata.artifact_id,
                artifact_sha256: metadata.artifact_sha256,
                initialization_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                runtime_version,
            },
            input_role: metadata.input_role,
            input_shape: metadata.input_shape,
            output_role: metadata.output_role,
            output_shape: metadata.output_shape,
        })
    }
}

impl RuntimeBackend for OpenVinoBackend {
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
                "OpenVINO requires a preprocessed NCHW tensor",
            ));
        };
        if role != self.input_role || shape != self.input_shape {
            return Err(InferenceError::new(
                "input_contract_mismatch",
                format!(
                    "expected role {} with shape {:?}, got role {role} with shape {shape:?}",
                    self.input_role, self.input_shape
                ),
            ));
        }
        if dtype != DType::F32 {
            return Err(InferenceError::new(
                "unsupported_dtype",
                "OpenVINO adapter requires f32 input",
            ));
        }
        let input_values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        let metadata = OpenVinoMetadata {
            platform: self.resolved.platform.clone(),
            model_version: self.resolved.model_version.clone(),
            artifact_id: self.resolved.artifact_id.clone(),
            artifact_sha256: self.resolved.artifact_sha256.clone(),
            input_role: self.input_role.clone(),
            input_shape: self.input_shape.clone(),
            output_role: self.output_role.clone(),
            output_shape: self.output_shape.clone(),
        };
        run_inference(&mut self.request, &metadata, &input_values)
    }
}

fn validate_metadata(model_bytes: &[u8], metadata: &OpenVinoMetadata) -> Result<(), InitFailure> {
    if metadata.platform != Platform::new("linux", "x86_64")
        || sha256_hex(model_bytes) != metadata.artifact_sha256
    {
        return Err(init_failure(
            metadata,
            "artifact_integrity_or_target_mismatch",
            InitializationStage::ArtifactIntegrity,
            "OpenVINO model bytes or target do not match the manifest",
        ));
    }
    element_count(&metadata.input_shape).map_err(|error| {
        init_failure(
            metadata,
            "openvino_io_contract_invalid",
            InitializationStage::IoDiscovery,
            error.to_string(),
        )
    })?;
    element_count(&metadata.output_shape).map_err(|error| {
        init_failure(
            metadata,
            "openvino_io_contract_invalid",
            InitializationStage::IoDiscovery,
            error.to_string(),
        )
    })?;
    Ok(())
}

fn validate_model_io(
    model: &openvino::Model,
    metadata: &OpenVinoMetadata,
) -> Result<(), InitFailure> {
    let input = model.get_input_by_index(0).map_err(|error| {
        init_failure(
            metadata,
            "openvino_io_discovery_failed",
            InitializationStage::IoDiscovery,
            error.to_string(),
        )
    })?;
    let output = model.get_output_by_index(0).map_err(|error| {
        init_failure(
            metadata,
            "openvino_io_discovery_failed",
            InitializationStage::IoDiscovery,
            error.to_string(),
        )
    })?;
    let discovered = (
        model.get_inputs_len(),
        model.get_outputs_len(),
        input.get_name(),
        output.get_name(),
        input.get_element_type(),
        output.get_element_type(),
        input.get_shape(),
        output.get_shape(),
    );
    let (
        Ok(1),
        Ok(1),
        Ok(input_name),
        Ok(output_name),
        Ok(ElementType::F32),
        Ok(ElementType::F32),
        Ok(input_shape),
        Ok(output_shape),
    ) = discovered
    else {
        return Err(init_failure(
            metadata,
            "openvino_io_contract_mismatch",
            InitializationStage::IoDiscovery,
            "OpenVINO model must expose one f32 input and one f32 output",
        ));
    };
    if input_name != "images"
        || output_name != "output0"
        || dimensions(&input_shape) != metadata.input_shape
        || dimensions(&output_shape) != metadata.output_shape
    {
        return Err(init_failure(
            metadata,
            "openvino_io_contract_mismatch",
            InitializationStage::IoDiscovery,
            format!(
                "discovered {input_name} {:?} -> {output_name} {:?}",
                input_shape.get_dimensions(),
                output_shape.get_dimensions()
            ),
        ));
    }
    Ok(())
}

fn run_inference(
    request: &mut InferRequest,
    metadata: &OpenVinoMetadata,
    input_values: &[f32],
) -> Result<RawModelOutput, InferenceError> {
    let shape = Shape::new(&usize_to_i64(&metadata.input_shape)?)
        .map_err(|error| inference_failure("openvino_input_prepare_failed", error))?;
    let mut tensor = Tensor::new(ElementType::F32, &shape)
        .map_err(|error| inference_failure("openvino_input_prepare_failed", error))?;
    tensor
        .get_data_mut::<f32>()
        .map_err(|error| inference_failure("openvino_input_prepare_failed", error))?
        .copy_from_slice(input_values);
    request
        .set_input_tensor_by_index(0, &tensor)
        .map_err(|error| inference_failure("openvino_input_prepare_failed", error))?;
    request
        .infer()
        .map_err(|error| inference_failure("openvino_inference_failed", error))?;
    let output = request
        .get_output_tensor_by_index(0)
        .map_err(|error| inference_failure("openvino_output_read_failed", error))?;
    let actual_shape = dimensions(
        &output
            .get_shape()
            .map_err(|error| inference_failure("openvino_output_read_failed", error))?,
    );
    let values = output
        .get_data::<f32>()
        .map_err(|error| inference_failure("openvino_output_read_failed", error))?
        .to_vec();
    if actual_shape != metadata.output_shape || values.iter().any(|value| !value.is_finite()) {
        return Err(InferenceError::new(
            "openvino_output_contract_mismatch",
            format!("expected {:?}, got {actual_shape:?}", metadata.output_shape),
        ));
    }
    Ok(RawModelOutput {
        tensors: vec![RawTensor {
            role: metadata.output_role.clone(),
            shape: actual_shape,
            data: TensorData::F32(values),
        }],
    })
}

fn dimensions(shape: &Shape) -> Vec<usize> {
    shape
        .get_dimensions()
        .iter()
        .filter_map(|dimension| usize::try_from(*dimension).ok())
        .collect()
}

fn usize_to_i64(shape: &[usize]) -> Result<Vec<i64>, InferenceError> {
    shape
        .iter()
        .map(|dimension| {
            i64::try_from(*dimension).map_err(|_| {
                InferenceError::new("input_shape_overflow", "input dimension exceeds i64")
            })
        })
        .collect()
}

fn element_count(shape: &[usize]) -> Result<usize, InferenceError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(InferenceError::new(
            "invalid_tensor_shape",
            "positive static shape is required",
        ));
    }
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| InferenceError::new("input_size_overflow", "tensor shape overflows"))
    })
}

fn init_failure(
    metadata: &OpenVinoMetadata,
    code: &'static str,
    stage: InitializationStage,
    message: impl Into<String>,
) -> InitFailure {
    InitFailure::new(code, stage, message).with_context(
        metadata.platform.clone(),
        metadata.model_version.clone(),
        BackendKind::OpenVino,
    )
}

fn inference_failure(code: &'static str, error: impl ToString) -> InferenceError {
    InferenceError::new(code, error.to_string())
}
