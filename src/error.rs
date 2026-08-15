//! Structured initialization and inference errors.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::backend::{BackendKind, Platform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutBoundary {
    NativeInitialization,
    WebInitialization,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitFailure {
    pub code: Box<str>,
    pub stage: InitializationStage,
    pub message: Box<str>,
    pub platform: Option<Box<Platform>>,
    pub model_version: Option<Box<str>>,
    pub attempted_backend: Option<BackendKind>,
}

impl InitFailure {
    pub fn new(
        code: impl Into<String>,
        stage: InitializationStage,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into().into_boxed_str(),
            stage,
            message: message.into().into_boxed_str(),
            platform: None,
            model_version: None,
            attempted_backend: None,
        }
    }

    pub fn with_context(
        mut self,
        platform: Platform,
        model_version: impl Into<String>,
        attempted_backend: BackendKind,
    ) -> Self {
        self.platform = Some(Box::new(platform));
        self.model_version = Some(model_version.into().into_boxed_str());
        self.attempted_backend = Some(attempted_backend);
        self
    }
}

impl fmt::Display for InitFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {:?}: {}",
            self.code, self.stage, self.message
        )
    }
}

impl std::error::Error for InitFailure {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceError {
    pub code: String,
    pub message: String,
}

impl InferenceError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for InferenceError {}
