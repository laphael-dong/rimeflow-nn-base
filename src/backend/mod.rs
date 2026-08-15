//! Stable backend vocabulary and injectable factory boundary.

use serde::{Deserialize, Serialize};

use crate::error::{InferenceError, InitFailure};

pub mod conformance;
pub mod coreml;

pub mod litert_v2;

#[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
pub mod legacy_ort;

pub use conformance::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, CapabilityStatus,
    ConformanceEvidenceKind, ConformanceReportError, ConformanceRunner, NativeAdapterCapability,
    OneShotNativeAdapterFactory, PlatformAdapterFactory, SelectedNativeAdapter,
    ADAPTER_CONFORMANCE_SCHEMA_V1, ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};
pub use coreml::{
    coreml_package_tree_sha256, CoreMlBackend, CoreMlIoMapping, CoreMlPackageIdentity,
    DEFAULT_COREML_INITIALIZATION_TIMEOUT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    LegacyOrt,
    CoreMl,
    LiteRtV2,
    WindowsMl,
    LinuxOrt,
    MindSporeLite,
    WebOnnx,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Platform {
    pub os: String,
    pub arch: String,
}

impl Platform {
    pub fn new(os: impl Into<String>, arch: impl Into<String>) -> Self {
        Self {
            os: os.into(),
            arch: arch.into(),
        }
    }

    pub fn current() -> Self {
        Self::new(std::env::consts::OS, std::env::consts::ARCH)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPlan {
    Full,
    Partitioned,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedBackend {
    pub backend_kind: BackendKind,
    pub platform: Platform,
    pub configured_provider: Option<String>,
    pub accelerator: Option<String>,
    pub execution_plan: ExecutionPlan,
    pub model_version: String,
    pub artifact_id: String,
    pub artifact_sha256: String,
    pub initialization_ms: u64,
    pub runtime_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    I8,
    U8,
    I32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ModelInput {
    Rgba8 {
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
    Tensor {
        role: String,
        shape: Vec<usize>,
        dtype: DType,
        bytes: Vec<u8>,
    },
}

impl ModelInput {
    pub fn validate(&self) -> Result<(), InferenceError> {
        match self {
            Self::Rgba8 {
                width,
                height,
                bytes,
            } => {
                let expected = (*width as usize)
                    .checked_mul(*height as usize)
                    .and_then(|pixels| pixels.checked_mul(4))
                    .ok_or_else(|| {
                        InferenceError::new("input_size_overflow", "RGBA dimensions overflow")
                    })?;
                if *width == 0 || *height == 0 || bytes.len() != expected {
                    return Err(InferenceError::new(
                        "invalid_rgba_shape",
                        format!("expected {expected} bytes, got {}", bytes.len()),
                    ));
                }
            }
            Self::Tensor {
                role,
                shape,
                dtype,
                bytes,
            } => {
                if role.is_empty() || shape.is_empty() || shape.contains(&0) {
                    return Err(InferenceError::new(
                        "invalid_tensor_shape",
                        "tensor role and positive static shape are required",
                    ));
                }
                let elements = shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
                let element_size = match dtype {
                    DType::F32 | DType::I32 => 4,
                    DType::F16 => 2,
                    DType::I8 | DType::U8 | DType::Bool => 1,
                    DType::I64 => 8,
                };
                let expected = elements
                    .and_then(|count| count.checked_mul(element_size))
                    .ok_or_else(|| {
                        InferenceError::new("input_size_overflow", "tensor shape overflows")
                    })?;
                if bytes.len() != expected {
                    return Err(InferenceError::new(
                        "invalid_tensor_bytes",
                        format!("expected {expected} bytes, got {}", bytes.len()),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "dtype", content = "values", rename_all = "lowercase")]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::I8(values) => values.len(),
            Self::U8(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawTensor {
    pub role: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawModelOutput {
    pub tensors: Vec<RawTensor>,
}

pub trait RuntimeBackend: Send {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendInitRequest {
    pub target: Platform,
    pub model_id: String,
    pub model_version: String,
    pub artifact_id: String,
    pub artifact_sha256: String,
}

pub struct BackendInstance<B> {
    pub backend: B,
    pub resolved: ResolvedBackend,
}

pub trait BackendFactory<B>: Send + Sync {
    fn create(&self, request: &BackendInitRequest) -> Result<BackendInstance<B>, InitFailure>;
}

impl<B, F> BackendFactory<B> for F
where
    F: Fn(&BackendInitRequest) -> Result<BackendInstance<B>, InitFailure> + Send + Sync,
{
    fn create(&self, request: &BackendInitRequest) -> Result<BackendInstance<B>, InitFailure> {
        self(request)
    }
}
