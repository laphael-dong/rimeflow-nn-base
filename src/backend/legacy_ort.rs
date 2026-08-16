//! Compatibility adapter around [`crate::native_ort::NativeOrtBackend`].

#![cfg(all(not(target_arch = "wasm32"), feature = "native"))]

use std::borrow::Cow;
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
        Self::from_model_bytes_with_kind(model_bytes, dst_size, metadata, BackendKind::LegacyOrt)
    }

    fn from_model_bytes_with_kind(
        model_bytes: &[u8],
        dst_size: u32,
        metadata: LegacyOrtMetadata,
        backend_kind: BackendKind,
    ) -> Result<Self, InitFailure> {
        if sha256_hex(model_bytes) != metadata.artifact_sha256 {
            return Err(InitFailure::new(
                "artifact_integrity_or_target_mismatch",
                InitializationStage::ArtifactIntegrity,
                "Legacy ORT model bytes do not match the manifest digest",
            )
            .with_context(metadata.platform, metadata.model_version, backend_kind));
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
                backend_kind,
            )
        })?;
        let resolved = ResolvedBackend {
            backend_kind,
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
        if values.len() != expected || !all_finite(&values) {
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

/// Linux ORT adapter with diagnostics distinct from the legacy compatibility path.
pub struct LinuxOrtBackend {
    inner: LegacyOrtBackend,
}

impl LinuxOrtBackend {
    pub fn from_model_bytes(
        model_bytes: &[u8],
        dst_size: u32,
        metadata: LegacyOrtMetadata,
    ) -> Result<Self, InitFailure> {
        LegacyOrtBackend::from_model_bytes_with_kind(
            model_bytes,
            dst_size,
            metadata,
            BackendKind::LinuxOrt,
        )
        .map(|inner| Self { inner })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        self.inner.resolved_backend()
    }

    pub fn infer_from_host_slice(
        &mut self,
        nchw: &[f32],
    ) -> Result<RawModelOutput, InferenceError> {
        self.inner.infer_from_host_slice(nchw)
    }
}

impl RuntimeBackend for LinuxOrtBackend {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        self.inner.infer(input)
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
        let values = f32_values_from_le_bytes(&bytes);
        self.infer_from_host_slice(&values)
    }
}

fn f32_values_from_le_bytes(bytes: &[u8]) -> Cow<'_, [f32]> {
    #[cfg(target_endian = "little")]
    if let Ok(values) = bytemuck::try_cast_slice(bytes) {
        return Cow::Borrowed(values);
    }

    Cow::Owned(
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect(),
    )
}

fn all_finite(values: &[f32]) -> bool {
    let mut index = 0;
    while index + 8 <= values.len() {
        if !values[index].is_finite()
            || !values[index + 1].is_finite()
            || !values[index + 2].is_finite()
            || !values[index + 3].is_finite()
            || !values[index + 4].is_finite()
            || !values[index + 5].is_finite()
            || !values[index + 6].is_finite()
            || !values[index + 7].is_finite()
        {
            return false;
        }
        index += 8;
    }
    values[index..].iter().all(|value| value.is_finite())
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

#[cfg(test)]
mod tests {
    use super::{all_finite, f32_values_from_le_bytes};
    use std::borrow::Cow;

    #[test]
    fn aligned_little_endian_f32_bytes_are_borrowed() {
        let values = [1.25f32, -3.5, 0.0];
        let decoded = f32_values_from_le_bytes(bytemuck::cast_slice(&values));

        assert_eq!(decoded.as_ref(), values);
        #[cfg(target_endian = "little")]
        assert!(matches!(decoded, Cow::Borrowed(_)));
    }

    #[test]
    fn unaligned_f32_bytes_preserve_little_endian_values() {
        let expected = [1.25f32, -3.5, 0.0];
        let mut storage = vec![0u8; expected.len() * 4 + 4];
        let aligned_offset = storage.as_ptr().align_offset(std::mem::align_of::<f32>());
        let unaligned_offset = (aligned_offset + 1) % 4;
        let bytes = &mut storage[unaligned_offset..unaligned_offset + expected.len() * 4];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip(expected) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        let decoded = f32_values_from_le_bytes(bytes);

        assert_eq!(decoded.as_ref(), expected);
        assert!(matches!(decoded, Cow::Owned(_)));
    }

    #[test]
    fn finite_scan_checks_every_unrolled_lane_and_remainder() {
        for index in 0..17 {
            let mut values = vec![1.0f32; 17];
            assert!(all_finite(&values));
            values[index] = f32::INFINITY;
            assert!(!all_finite(&values), "missed non-finite value at {index}");
        }
        assert!(all_finite(&[]));
    }
}
