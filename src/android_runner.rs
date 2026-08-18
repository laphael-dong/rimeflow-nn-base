//! Android LiteRT runner 的可移植 bundle 与机器报告合约。

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backend::{DType, Platform, ResolvedBackend};

pub const ANDROID_RUNNER_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const ANDROID_RUNNER_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerFile {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl AndroidRunnerFile {
    pub fn resolve_and_verify(&self, root: &Path) -> Result<PathBuf, String> {
        validate_relative_path(&self.path)?;
        validate_sha256(&self.sha256)?;
        let path = root.join(&self.path);
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("无法读取 bundle 文件 {}: {error}", path.display()))?;
        if bytes.len() as u64 != self.bytes {
            return Err(format!(
                "bundle 文件 {} 长度不符：预期 {}，实际 {}",
                self.path,
                self.bytes,
                bytes.len()
            ));
        }
        let actual = sha256_hex(&bytes);
        if actual != self.sha256 {
            return Err(format!(
                "bundle 文件 {} SHA-256 不符：预期 {}，实际 {actual}",
                self.path, self.sha256
            ));
        }
        Ok(path)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AndroidRunnerFixtureKind {
    Golden,
    FaultNeverPromote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerFixture {
    pub id: String,
    pub kind: AndroidRunnerFixtureKind,
    pub input: AndroidRunnerFile,
    pub role: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerGates {
    pub initialization_deadline_ms: u64,
    pub golden_runs: u32,
    pub performance_warmup_runs: u32,
    pub performance_sample_runs: u32,
    pub collect_peak_rss: bool,
    pub collect_package_load: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidBundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub target: Platform,
    pub minimum_api: u32,
    pub cpu_only: bool,
    pub runner: AndroidRunnerFile,
    pub runtime_libraries: Vec<AndroidRunnerFile>,
    pub provenance: Vec<AndroidRunnerFile>,
    pub model_manifest: AndroidRunnerFile,
    pub artifact: AndroidRunnerFile,
    pub fixtures: Vec<AndroidRunnerFixture>,
    pub gates: AndroidRunnerGates,
}

impl AndroidBundleManifest {
    pub fn parse_and_validate(json: &str) -> Result<Self, String> {
        let value: Self = serde_json::from_str(json)
            .map_err(|error| format!("bundle manifest JSON 无效: {error}"))?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != ANDROID_RUNNER_BUNDLE_SCHEMA_VERSION {
            return Err(format!(
                "不支持 bundle schemaVersion {}",
                self.schema_version
            ));
        }
        if self.bundle_id.is_empty()
            || ![
                Platform::new("android", "arm64"),
                Platform::new("android", "x86_64"),
            ]
            .contains(&self.target)
        {
            return Err(
                "bundle 必须具有非空 ID，且目标必须是 android/arm64 或 android/x86_64".to_owned(),
            );
        }
        if self.minimum_api < 26 || !self.cpu_only {
            return Err("bundle 仅允许 Android API 26+ 的 CPU LiteRT".to_owned());
        }
        if self.runtime_libraries.len() != 1
            || !self
                .runtime_libraries
                .iter()
                .any(|file| file.path.ends_with("/libLiteRt.so") || file.path == "lib/libLiteRt.so")
        {
            return Err("bundle 必须且只能明确包含一个 lib/libLiteRt.so".to_owned());
        }
        if self.fixtures.is_empty() {
            return Err("bundle 至少需要一个外部输入 fixture".to_owned());
        }
        if self.gates.initialization_deadline_ms == 0
            || self.gates.golden_runs != 2
            || self.gates.performance_warmup_runs != 5
            || self.gates.performance_sample_runs == 0
            || !self.gates.collect_peak_rss
            || !self.gates.collect_package_load
        {
            return Err(
                "bundle gate 必须固定为双跑 golden、5 次 warmup、性能/RSS/package-load 采集"
                    .to_owned(),
            );
        }
        let mut ids = HashSet::new();
        for file in std::iter::once(&self.runner)
            .chain(&self.runtime_libraries)
            .chain(&self.provenance)
            .chain(std::iter::once(&self.model_manifest))
            .chain(std::iter::once(&self.artifact))
            .chain(self.fixtures.iter().map(|fixture| &fixture.input))
        {
            validate_relative_path(&file.path)?;
            validate_sha256(&file.sha256)?;
            if file.bytes == 0 {
                return Err(format!("bundle 文件 {} 的长度必须大于零", file.path));
            }
        }
        for fixture in &self.fixtures {
            if fixture.id.is_empty()
                || !ids.insert(fixture.id.as_str())
                || fixture.role.is_empty()
                || fixture.shape.is_empty()
                || fixture.shape.contains(&0)
            {
                return Err("fixture ID/role/shape 必须非空、唯一且为静态正维度".to_owned());
            }
        }
        Ok(())
    }

    pub fn validate_target_arch(&self, compiled_arch: &str) -> Result<(), String> {
        if self.target == Platform::new("android", compiled_arch) {
            Ok(())
        } else {
            Err(format!(
                "bundle target {}/{} 与 runner 编译架构 android/{compiled_arch} 不一致",
                self.target.os, self.target.arch
            ))
        }
    }

    pub fn verify_all_files(&self, root: &Path) -> Result<Vec<PathBuf>, String> {
        let files = std::iter::once(&self.runner)
            .chain(&self.runtime_libraries)
            .chain(&self.provenance)
            .chain(std::iter::once(&self.model_manifest))
            .chain(std::iter::once(&self.artifact))
            .chain(self.fixtures.iter().map(|fixture| &fixture.input));
        files.map(|file| file.resolve_and_verify(root)).collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AndroidRunnerReportState {
    RuntimeVerified,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerOutput {
    pub fixture_id: String,
    pub fixture_kind: AndroidRunnerFixtureKind,
    pub run: u32,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub inference_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerPerformance {
    pub initialization_ms: u64,
    pub initialization_deadline_ms: u64,
    pub cold_inference_ms: Option<f64>,
    pub warmup_runs: u32,
    pub warm_inference_ms: Vec<f64>,
    pub peak_process_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerDevice {
    pub serial: Option<String>,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub fingerprint: Option<String>,
    pub api_level: Option<String>,
    pub abi: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidLoadedLibrary {
    pub path: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AndroidRunnerReport {
    pub schema_version: u32,
    pub state: AndroidRunnerReportState,
    pub selection_code: String,
    pub failure_stage: Option<String>,
    pub failure_message: Option<String>,
    pub bundle_id: String,
    pub bundle_manifest_sha256: String,
    pub target: Platform,
    pub runner_sha256: String,
    pub model_manifest_sha256: String,
    pub artifact_sha256: String,
    pub runtime_version: String,
    pub rust_binding_version: String,
    pub configured_provider: String,
    pub accelerator: String,
    pub resolved: Option<ResolvedBackend>,
    pub io_diagnostics: Option<serde_json::Value>,
    pub device: AndroidRunnerDevice,
    pub outputs: Vec<AndroidRunnerOutput>,
    pub performance: AndroidRunnerPerformance,
    pub loaded_libraries: Vec<AndroidLoadedLibrary>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!("无效的小写 SHA-256: {value}"))
    }
}

fn validate_relative_path(value: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        Err(format!("bundle 路径必须保持在根目录内: {value}"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> AndroidRunnerFile {
        AndroidRunnerFile {
            path: path.to_owned(),
            sha256: "a".repeat(64),
            bytes: 1,
        }
    }

    fn manifest() -> AndroidBundleManifest {
        AndroidBundleManifest {
            schema_version: 1,
            bundle_id: "sha256-deadbeef".to_owned(),
            target: Platform::new("android", "arm64"),
            minimum_api: 26,
            cpu_only: true,
            runner: file("bin/rimeflow-android-litert-runner"),
            runtime_libraries: vec![file("lib/libLiteRt.so")],
            provenance: vec![file("provenance/litert-artifact-manifest.json")],
            model_manifest: file("manifest/model-manifest.json"),
            artifact: file("model/yolov8n.tflite"),
            fixtures: vec![AndroidRunnerFixture {
                id: "single-target".to_owned(),
                kind: AndroidRunnerFixtureKind::Golden,
                input: file("inputs/single-target.f32le.bin"),
                role: "image".to_owned(),
                shape: vec![1, 3, 640, 640],
                dtype: DType::F32,
            }],
            gates: AndroidRunnerGates {
                initialization_deadline_ms: 30_000,
                golden_runs: 2,
                performance_warmup_runs: 5,
                performance_sample_runs: 30,
                collect_peak_rss: true,
                collect_package_load: true,
            },
        }
    }

    #[test]
    fn accepts_frozen_android_cpu_contract() {
        manifest().validate().unwrap();
        let mut x86 = manifest();
        x86.target = Platform::new("android", "x86_64");
        x86.validate().unwrap();
        x86.validate_target_arch("x86_64").unwrap();
        assert!(x86.validate_target_arch("arm64").is_err());

        let mut mixed = manifest();
        mixed.runtime_libraries.push(file("lib/arm64/libLiteRt.so"));
        assert!(mixed.validate().is_err());
    }

    #[test]
    fn rejects_path_escape_and_gpu_claim() {
        let mut value = manifest();
        value.artifact.path = "../model.tflite".to_owned();
        value.cpu_only = false;
        assert!(value.validate().is_err());
    }

    #[test]
    fn verifies_file_digest_before_use() {
        let root = std::env::temp_dir().join(format!("rfb-runner-contract-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("input.bin");
        std::fs::write(&path, b"verified").unwrap();
        let identity = AndroidRunnerFile {
            path: "input.bin".to_owned(),
            sha256: sha256_hex(b"verified"),
            bytes: 8,
        };
        assert_eq!(identity.resolve_and_verify(&root).unwrap(), path);
        let _ = std::fs::remove_dir_all(root);
    }
}
