#![cfg(not(any(target_os = "macos", target_os = "ios")))]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rimeflow_onnx_base::{
    coreml_package_tree_sha256, AdapterConformanceCase, AdapterConformanceCheck,
    AdapterConformanceCheckKind, AdapterConformanceReport, AdapterConformanceStatus,
    AdapterSelection, BackendInitRequest, BackendKind, ConformanceEvidenceKind, ConformanceRunner,
    CoreMlBackend, CoreMlIoMapping, InitFailure, InitializationStage, ModelManifest, Platform,
    ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};

const MANIFEST_JSON: &str = include_str!("fixtures/conformance/coreml-manifest.json");
const VALIDATION_TREE_SHA256: &str =
    "299e6218590fb62da49407e334431a43a999a96020bdf52b4ebc04708218fb98";
const SYNTHETIC_TREE_SHA256: &str =
    "b5c7cf2cf8eb1b6dc313874a84b0006e4b3ef778c0d18bbcb0f06b425fbfd562";

#[test]
fn manifest_maps_logical_roles_to_validation_feature_names() {
    let manifest = manifest();
    let mapping = CoreMlIoMapping::from_manifest(&manifest, &request(VALIDATION_TREE_SHA256))
        .expect("Validation Core ML I/O mapping");

    assert_eq!(mapping.input_role, "image");
    assert_eq!(mapping.input_feature_name, "image");
    assert_eq!((mapping.input_width, mapping.input_height), (640, 640));
    assert_eq!(mapping.output_role, "detections");
    assert_eq!(mapping.output_feature_name, "var_911");
    assert_eq!(mapping.output_shape, [1, 84, 8400]);
}

#[test]
fn package_tree_identity_matches_validation_canonicalization() {
    let package = SyntheticPackage::new();
    let identity = coreml_package_tree_sha256(package.path()).expect("package tree identity");

    assert_eq!(identity.tree_sha256, SYNTHETIC_TREE_SHA256);
    assert_eq!(identity.file_count, 3);
    assert_eq!(identity.total_file_bytes, 16);
}

#[test]
fn non_apple_target_returns_structured_coreml_fallback_after_identity_check() {
    let package = SyntheticPackage::new();
    let manifest = synthetic_manifest();
    let error = match CoreMlBackend::load_package(
        package.path(),
        &manifest,
        &request(SYNTHETIC_TREE_SHA256),
        Duration::from_secs(1),
    ) {
        Ok(_) => panic!("Linux must not claim Apple Core ML readiness"),
        Err(error) => error,
    };

    assert_eq!(error.code.as_ref(), "coreml_target_unavailable");
    assert_eq!(error.stage, InitializationStage::RuntimeLoad);
    assert_eq!(error.attempted_backend, Some(BackendKind::CoreMl));
    assert_eq!(
        error.platform.as_deref(),
        Some(&Platform::new("macos", "aarch64"))
    );
}

#[test]
fn package_identity_drift_fails_before_runtime_selection() {
    let package = SyntheticPackage::new();
    let manifest = synthetic_manifest();
    let error = match CoreMlBackend::load_package(
        package.path(),
        &manifest,
        &request(&"0".repeat(64)),
        Duration::from_secs(1),
    ) {
        Ok(_) => panic!("mismatched identity must fail"),
        Err(error) => error,
    };

    assert_eq!(error.code.as_ref(), "artifact_integrity_or_target_mismatch");
    assert_eq!(error.stage, InitializationStage::ArtifactIntegrity);
}

#[test]
fn model_identity_drift_fails_before_runtime_selection() {
    let mut manifest = synthetic_manifest();
    manifest.model.version = "different-model-version".to_owned();
    let error = CoreMlIoMapping::from_manifest(&manifest, &request(SYNTHETIC_TREE_SHA256))
        .expect_err("mismatched model identity must fail");

    assert_eq!(error.code.as_ref(), "artifact_integrity_or_target_mismatch");
    assert_eq!(error.stage, InitializationStage::ArtifactIntegrity);
}

#[test]
fn linux_conformance_report_is_blocked_without_an_apple_runner() {
    let request = request(VALIDATION_TREE_SHA256);
    let failure = InitFailure::new(
        "coreml_target_unavailable",
        InitializationStage::RuntimeLoad,
        "no real macOS or iOS Core ML runner is attached",
    )
    .with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::CoreMl,
    );
    let report = AdapterConformanceReport {
        schema_version: ADAPTER_CONFORMANCE_SCHEMA_VERSION,
        case: AdapterConformanceCase {
            id: "macos-aarch64-coreml".to_owned(),
            model_id: request.model_id.clone(),
            model_version: request.model_version.clone(),
            target: request.target.clone(),
            adapter: BackendKind::CoreMl,
            artifact_id: request.artifact_id.clone(),
            artifact_sha256: request.artifact_sha256.clone(),
            manifest_sha256: "b15827142c2f69649489aeaa8877e18ea1f4e5dc3e342d04c772fad78c6c5f40"
                .to_owned(),
            native_initialization_timeout_ms: 15_000,
        },
        runner: ConformanceRunner {
            kind: ConformanceEvidenceKind::BuildOnly,
            target: request.target,
            runner_id: None,
        },
        selection: AdapterSelection::UseWebFallback { failure },
        checks: AdapterConformanceCheckKind::ALL
            .iter()
            .copied()
            .map(|kind| {
                let build_verified = matches!(
                    kind,
                    AdapterConformanceCheckKind::ManifestIo
                        | AdapterConformanceCheckKind::InitializationTimeout
                        | AdapterConformanceCheckKind::FaultInjection
                        | AdapterConformanceCheckKind::Diagnostics
                );
                AdapterConformanceCheck {
                    kind,
                    status: if build_verified {
                        AdapterConformanceStatus::BuildVerified
                    } else {
                        AdapterConformanceStatus::Blocked
                    },
                    detail: if build_verified {
                        "covered by Linux tests and Apple-target static compilation".to_owned()
                    } else {
                        "requires a real macOS or iOS Core ML runner".to_owned()
                    },
                    evidence_path: build_verified.then(|| "tests/coreml_adapter.rs".to_owned()),
                }
            })
            .collect(),
    };

    report.validate().expect("honest build-only report");
    assert_eq!(report.overall_status(), AdapterConformanceStatus::Blocked);
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    AdapterConformanceReport::parse_and_validate(&json).expect("round trip report");
}

#[test]
fn machine_report_does_not_claim_apple_support_or_runtime_execution() {
    let report: serde_json::Value = serde_json::from_str(include_str!(
        "../reports/os6-base-coreml-adapter-implementation-report.json"
    ))
    .expect("machine report JSON");

    assert_eq!(report["status"], "build-verified-apple-runner-blocked");
    assert_eq!(
        report["completionClaims"]["realAppleRuntimeExecuted"],
        false
    );
    assert_eq!(report["completionClaims"]["supportedPlatform"], false);
    assert_eq!(report["completionClaims"]["performance"], false);
    assert_eq!(report["completionClaims"]["productPackage"], false);
}

fn manifest() -> ModelManifest {
    ModelManifest::parse_and_validate(MANIFEST_JSON).expect("Core ML manifest fixture")
}

fn synthetic_manifest() -> ModelManifest {
    let mut value: serde_json::Value = serde_json::from_str(MANIFEST_JSON).expect("manifest JSON");
    value["artifacts"][0]["sha256"] = serde_json::json!(SYNTHETIC_TREE_SHA256);
    ModelManifest::parse_and_validate(&value.to_string()).expect("synthetic Core ML manifest")
}

fn request(sha256: &str) -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("macos", "aarch64"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-coreml-20260811".to_owned(),
        artifact_id: "apple-coreml-fp32".to_owned(),
        artifact_sha256: sha256.to_owned(),
    }
}

struct SyntheticPackage {
    root: PathBuf,
}

impl SyntheticPackage {
    fn new() -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rimeflow-coreml-package-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("Data/com.apple.CoreML/weights"))
            .expect("create synthetic package directories");
        fs::write(root.join("Data/com.apple.CoreML/model.mlmodel"), b"model\n")
            .expect("write synthetic model");
        fs::write(
            root.join("Data/com.apple.CoreML/weights/weight.bin"),
            b"weight\n",
        )
        .expect("write synthetic weights");
        fs::write(root.join("Manifest.json"), b"{}\n").expect("write synthetic manifest");
        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for SyntheticPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
