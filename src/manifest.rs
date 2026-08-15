//! Versioned model manifest types and schema/semantic validation.

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backend::{DType, Platform};

pub const MODEL_MANIFEST_SCHEMA_V1: &str = include_str!("../schemas/model-manifest.schema.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model: ModelIdentity,
    pub tensors: TensorGroups,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelIdentity {
    pub id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorGroups {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorSpec {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub shape: Vec<i64>,
    pub layout: Layout,
    pub dtype: DType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    #[serde(rename = "NCHW")]
    Nchw,
    #[serde(rename = "NHWC")]
    Nhwc,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Quantization {
    pub scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_point: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Artifact {
    pub id: String,
    pub format: ArtifactFormat,
    pub targets: Vec<ArtifactTarget>,
    pub path: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub converter: Option<Converter>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactFormat {
    Onnx,
    Coreml,
    Tflite,
    Windowsml,
    MindsporeLite,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactTarget {
    pub os: String,
    pub arch: String,
}

impl ArtifactTarget {
    pub fn matches(&self, target: &Platform) -> bool {
        self.os == target.os && self.arch == target.arch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Converter {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    SchemaInvalid(String),
    SchemaVersionUnsupported(u32),
    QuantizationZeroPointMissing(String),
    QuantizationInvalid(String),
    StaticShapeInvalid(String),
    RoleInvalid(String),
    ArtifactIntegrityOrTargetMismatch(String),
}

impl ManifestError {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::SchemaInvalid(_) => "manifest_schema_invalid",
            Self::SchemaVersionUnsupported(_) => "manifest_schema_version_unsupported",
            Self::QuantizationZeroPointMissing(_) => "quantization_zero_point_missing",
            Self::QuantizationInvalid(_) => "quantization_invalid",
            Self::StaticShapeInvalid(_) => "static_shape_invalid",
            Self::RoleInvalid(_) => "manifest_role_invalid",
            Self::ArtifactIntegrityOrTargetMismatch(_) => "artifact_integrity_or_target_mismatch",
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaInvalid(message) => write!(formatter, "invalid manifest schema: {message}"),
            Self::SchemaVersionUnsupported(version) => {
                write!(formatter, "unsupported manifest schema version {version}")
            }
            Self::QuantizationZeroPointMissing(role) => {
                write!(formatter, "quantized tensor {role} has no zeroPoint")
            }
            Self::QuantizationInvalid(role) => {
                write!(
                    formatter,
                    "tensor {role} has invalid quantization parameters"
                )
            }
            Self::StaticShapeInvalid(role) => {
                write!(formatter, "tensor {role} has an invalid static shape")
            }
            Self::RoleInvalid(message) => write!(formatter, "invalid tensor role: {message}"),
            Self::ArtifactIntegrityOrTargetMismatch(message) => {
                write!(
                    formatter,
                    "artifact integrity or target mismatch: {message}"
                )
            }
        }
    }
}

impl std::error::Error for ManifestError {}

impl ModelManifest {
    /// Validate the structural constraints represented by the checked-in v1
    /// JSON Schema. Cross-field semantic failures retain their own error codes.
    pub fn validate_schema_json(json: &str) -> Result<(), ManifestError> {
        let manifest = Self::decode(json)?;
        manifest.validate_schema_constraints()
    }

    pub fn parse_and_validate(json: &str) -> Result<Self, ManifestError> {
        let manifest = Self::decode(json)?;
        manifest.validate_semantics()?;
        Ok(manifest)
    }

    pub fn validate_semantics(&self) -> Result<(), ManifestError> {
        if self.schema_version != 1 {
            return Err(ManifestError::SchemaVersionUnsupported(self.schema_version));
        }
        self.validate_common_structure()?;

        let mut roles = HashSet::new();
        for tensor in self.tensors.inputs.iter().chain(&self.tensors.outputs) {
            if !roles.insert(tensor.role.as_str()) {
                return Err(ManifestError::RoleInvalid(format!(
                    "duplicate role {}",
                    tensor.role
                )));
            }
            match tensor.dtype {
                DType::I8 | DType::U8 => {
                    let quantization = tensor
                        .quantization
                        .as_ref()
                        .ok_or_else(|| ManifestError::QuantizationInvalid(tensor.role.clone()))?;
                    if quantization.zero_point.is_none() {
                        return Err(ManifestError::QuantizationZeroPointMissing(
                            tensor.role.clone(),
                        ));
                    }
                    if !quantization.scale.is_finite() || quantization.scale <= 0.0 {
                        return Err(ManifestError::QuantizationInvalid(tensor.role.clone()));
                    }
                    let zero_point = quantization.zero_point.expect("checked above");
                    let in_range = match tensor.dtype {
                        DType::I8 => (-128..=127).contains(&zero_point),
                        DType::U8 => (0..=255).contains(&zero_point),
                        _ => unreachable!(),
                    };
                    if !in_range {
                        return Err(ManifestError::QuantizationInvalid(tensor.role.clone()));
                    }
                }
                _ => {
                    if let Some(quantization) = &tensor.quantization {
                        if !quantization.scale.is_finite() || quantization.scale <= 0.0 {
                            return Err(ManifestError::QuantizationInvalid(tensor.role.clone()));
                        }
                    }
                }
            }
        }

        let input_roles: HashSet<_> = self
            .tensors
            .inputs
            .iter()
            .map(|tensor| tensor.role.as_str())
            .collect();
        let output_roles: HashSet<_> = self
            .tensors
            .outputs
            .iter()
            .map(|tensor| tensor.role.as_str())
            .collect();
        let mut artifact_ids = HashSet::new();
        for artifact in &self.artifacts {
            if !artifact_ids.insert(artifact.id.as_str()) {
                return Err(ManifestError::ArtifactIntegrityOrTargetMismatch(format!(
                    "duplicate artifact id {}",
                    artifact.id
                )));
            }
            if artifact.sha256.bytes().all(|byte| byte == b'0') {
                return Err(ManifestError::ArtifactIntegrityOrTargetMismatch(format!(
                    "artifact {} uses the all-zero digest",
                    artifact.id
                )));
            }
            if artifact
                .inputs
                .iter()
                .any(|role| !input_roles.contains(role.as_str()))
                || artifact
                    .outputs
                    .iter()
                    .any(|role| !output_roles.contains(role.as_str()))
            {
                return Err(ManifestError::RoleInvalid(format!(
                    "artifact {} references an unknown role",
                    artifact.id
                )));
            }
        }
        Ok(())
    }

    pub fn select_artifact(
        &self,
        artifact_id: &str,
        target: &Platform,
    ) -> Result<&Artifact, ManifestError> {
        self.validate_semantics()?;
        self.artifacts
            .iter()
            .find(|artifact| {
                artifact.id == artifact_id
                    && artifact
                        .targets
                        .iter()
                        .any(|candidate| candidate.matches(target))
            })
            .ok_or_else(|| {
                ManifestError::ArtifactIntegrityOrTargetMismatch(format!(
                    "artifact {artifact_id} does not target {}/{}",
                    target.os, target.arch
                ))
            })
    }

    pub fn verify_artifact_bytes(artifact: &Artifact, bytes: &[u8]) -> Result<(), ManifestError> {
        let actual = sha256_hex(bytes);
        if actual != artifact.sha256 {
            return Err(ManifestError::ArtifactIntegrityOrTargetMismatch(format!(
                "artifact {} expected {}, got {actual}",
                artifact.id, artifact.sha256
            )));
        }
        Ok(())
    }

    fn decode(json: &str) -> Result<Self, ManifestError> {
        serde_json::from_str(json).map_err(|error| ManifestError::SchemaInvalid(error.to_string()))
    }

    fn validate_schema_constraints(&self) -> Result<(), ManifestError> {
        if self.schema_version != 1 {
            return Err(ManifestError::SchemaInvalid(
                "schemaVersion must be 1".to_owned(),
            ));
        }
        self.validate_common_structure()?;
        for tensor in self.tensors.inputs.iter().chain(&self.tensors.outputs) {
            if matches!(tensor.dtype, DType::I8 | DType::U8) {
                let Some(quantization) = &tensor.quantization else {
                    return Err(ManifestError::SchemaInvalid(format!(
                        "quantized tensor {} requires quantization",
                        tensor.role
                    )));
                };
                if quantization.zero_point.is_none() {
                    return Err(ManifestError::SchemaInvalid(format!(
                        "quantized tensor {} requires zeroPoint",
                        tensor.role
                    )));
                }
            }
        }
        if self
            .artifacts
            .iter()
            .any(|artifact| artifact.sha256.bytes().all(|byte| byte == b'0'))
        {
            return Err(ManifestError::SchemaInvalid(
                "artifact digest must not be all zeroes".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_common_structure(&self) -> Result<(), ManifestError> {
        if self.model.id.is_empty()
            || self.model.version.is_empty()
            || self.tensors.inputs.is_empty()
            || self.tensors.outputs.is_empty()
            || self.artifacts.is_empty()
        {
            return Err(ManifestError::SchemaInvalid(
                "model, tensors, and artifacts must be non-empty".to_owned(),
            ));
        }
        for tensor in self.tensors.inputs.iter().chain(&self.tensors.outputs) {
            if tensor.role.is_empty()
                || tensor.name.as_ref().is_some_and(|name| name.is_empty())
                || (tensor.name.is_none() && tensor.index.is_none())
            {
                return Err(ManifestError::SchemaInvalid(
                    "tensor role and name or index are required".to_owned(),
                ));
            }
            if tensor.shape.is_empty() || tensor.shape.iter().any(|dimension| *dimension <= 0) {
                return Err(ManifestError::StaticShapeInvalid(tensor.role.clone()));
            }
        }
        for artifact in &self.artifacts {
            if artifact.id.is_empty()
                || artifact.path.is_empty()
                || artifact.targets.is_empty()
                || artifact.inputs.is_empty()
                || artifact.outputs.is_empty()
                || !valid_sha256(&artifact.sha256)
            {
                return Err(ManifestError::SchemaInvalid(format!(
                    "artifact {} is structurally incomplete",
                    artifact.id
                )));
            }
            if artifact
                .targets
                .iter()
                .any(|target| target.os.is_empty() || target.arch.is_empty())
            {
                return Err(ManifestError::SchemaInvalid(format!(
                    "artifact {} contains an empty target",
                    artifact.id
                )));
            }
            if artifact
                .converter
                .as_ref()
                .is_some_and(|converter| converter.name.is_empty() || converter.version.is_empty())
            {
                return Err(ManifestError::SchemaInvalid(format!(
                    "artifact {} has an incomplete converter",
                    artifact.id
                )));
            }
        }
        Ok(())
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
