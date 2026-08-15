//! Compatibility adapter around [`crate::native_ort::NativeOrtBackend`].

#![cfg(all(not(target_arch = "wasm32"), feature = "native"))]

use std::time::Instant;

use crate::backend::{
    BackendKind, DType, ExecutionPlan, ModelInput, Platform, RawModelOutput, RawTensor,
    ResolvedBackend, RuntimeBackend, TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::sha256_hex;
use crate::native_ort::{InferError, NativeOrtBackend, ResolvedEp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyOrtMetadata {
    pub platform: Platform,
    pub model_version: String,
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub output_role: String,
    pub output_shape: Vec<usize>,
    pub runtime_version: Option<String>,
}

pub struct LegacyOrtBackend {
    inner: NativeOrtBackend,
    resolved: ResolvedBackend,
    output_role: String,
    output_shape: Vec<usize>,
}

impl LegacyOrtBackend {
    pub fn from_model_bytes(
        model_bytes: &[u8],
        dst_size: u32,
        metadata: LegacyOrtMetadata,
    ) -> Result<Self, InitFailure> {
        if sha256_hex(model_bytes) != metadata.artifact_sha256 {
            return Err(InitFailure::new(
                "artifact_integrity_or_target_mismatch",
                InitializationStage::ArtifactIntegrity,
                "Legacy ORT model bytes do not match the manifest digest",
            )
            .with_context(
                metadata.platform,
                metadata.model_version,
                BackendKind::LegacyOrt,
            ));
        }
        let started = Instant::now();
        let inner = NativeOrtBackend::from_model_bytes(model_bytes, dst_size).map_err(|error| {
            InitFailure::new(
                "native_initialization_failed",
                InitializationStage::ModelCompile,
                error.to_string(),
            )
            .with_context(
                metadata.platform.clone(),
                metadata.model_version.clone(),
                BackendKind::LegacyOrt,
            )
        })?;
        let resolved = ResolvedBackend {
            backend_kind: BackendKind::LegacyOrt,
            platform: metadata.platform,
            configured_provider: Some(configured_provider(inner.resolved_ep()).to_owned()),
            accelerator: None,
            execution_plan: ExecutionPlan::Unknown,
            model_version: metadata.model_version,
            artifact_id: metadata.artifact_id,
            artifact_sha256: metadata.artifact_sha256,
            initialization_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            runtime_version: metadata.runtime_version,
        };
        Ok(Self {
            inner,
            resolved,
            output_role: metadata.output_role,
            output_shape: metadata.output_shape,
        })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.resolved
    }

    pub fn infer_from_host_slice(
        &mut self,
        nchw: &[f32],
    ) -> Result<RawModelOutput, InferenceError> {
        let values = self
            .inner
            .infer_from_host_slice(nchw)
            .map_err(map_inference_error)?;
        let expected = self
            .output_shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| {
                InferenceError::new("output_shape_overflow", "output shape overflows")
            })?;
        if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
            return Err(InferenceError::new(
                "smoke_output_invalid",
                format!("expected {expected} finite values, got {}", values.len()),
            ));
        }
        Ok(RawModelOutput {
            tensors: vec![RawTensor {
                role: self.output_role.clone(),
                shape: self.output_shape.clone(),
                data: TensorData::F32(values),
            }],
        })
    }
}

impl RuntimeBackend for LegacyOrtBackend {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        input.validate()?;
        let ModelInput::Tensor { dtype, bytes, .. } = input else {
            return Err(InferenceError::new(
                "unsupported_input",
                "Legacy ORT requires a preprocessed NCHW tensor",
            ));
        };
        if dtype != DType::F32 {
            return Err(InferenceError::new(
                "unsupported_dtype",
                "Legacy ORT fast path requires f32 input",
            ));
        }
        let mut values = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes(
                chunk.try_into().expect("four-byte chunk"),
            ));
        }
        self.infer_from_host_slice(&values)
    }
}

fn configured_provider(provider: ResolvedEp) -> &'static str {
    match provider {
        ResolvedEp::CoreML => "CoreML",
        ResolvedEp::DirectML => "DirectML",
        ResolvedEp::Cuda => "CUDA",
        ResolvedEp::TensorRt => "TensorRT",
        ResolvedEp::Nnapi => "NNAPI",
        ResolvedEp::Qnn => "QNN",
        ResolvedEp::OpenVino => "OpenVINO",
        ResolvedEp::Cpu => "CPU",
    }
}

fn map_inference_error(error: InferError) -> InferenceError {
    InferenceError::new("inference_failed", error.to_string())
}
