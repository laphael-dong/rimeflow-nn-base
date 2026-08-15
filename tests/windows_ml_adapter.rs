use std::time::Duration;

use rimeflow_onnx_base::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, BackendInitRequest,
    BackendKind, ConformanceEvidenceKind, ConformanceRunner, InitFailure, InitializationStage,
    ModelManifest, Platform, RuntimeLifecycle, WindowsMlAdapterConfig, WindowsMlAdapterFactory,
    WindowsMlMachineReport, WindowsMlRoleMap, WindowsMlRunnerCommand,
    ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};

const MODEL_SHA256: &str = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad";
const MANIFEST_SHA256: &str = "6f411aedec1550f3306459468dc3b4a0a4bc2558f5233f5f25404f1ac50e9c26";

#[test]
fn canonical_onnx_manifest_roles_map_to_windows_ml_features() {
    let manifest = manifest("onnx");
    let roles = WindowsMlRoleMap::from_manifest(&manifest, &request())
        .expect("canonical ONNX is a Windows ML input artifact");

    assert_eq!(roles.input.role, "image");
    assert_eq!(roles.input.name.as_deref(), Some("images"));
    assert_eq!(roles.input.shape, [1, 3, 640, 640]);
    assert_eq!(roles.outputs.len(), 1);
    assert_eq!(roles.outputs[0].role, "detections");
    assert_eq!(roles.outputs[0].name.as_deref(), Some("output0"));
    assert_eq!(roles.outputs[0].shape, [1, 84, 8400]);
}

#[test]
fn role_mapping_rejects_multi_input_and_non_f32_contracts() {
    let mut value = manifest_value("onnx");
    value["tensors"]["inputs"]
        .as_array_mut()
        .expect("inputs")
        .push(serde_json::json!({
            "role": "metadata", "name": "metadata", "shape": [1],
            "layout": "NCHW", "dtype": "f32"
        }));
    value["artifacts"][0]["inputs"] = serde_json::json!(["image", "metadata"]);
    let manifest = ModelManifest::parse_and_validate(&value.to_string()).expect("valid manifest");
    let error = WindowsMlRoleMap::from_manifest(&manifest, &request())
        .expect_err("base adapter cannot represent multiple ModelInput values");
    assert_eq!(error.code.as_ref(), "windows_ml_multi_input_unsupported");
    assert_eq!(error.stage, InitializationStage::IoDiscovery);

    let mut value = manifest_value("onnx");
    value["tensors"]["inputs"][0]["dtype"] = serde_json::json!("i32");
    let manifest = ModelManifest::parse_and_validate(&value.to_string()).expect("valid manifest");
    let error = WindowsMlRoleMap::from_manifest(&manifest, &request())
        .expect_err("runner currently accepts only the frozen f32 contract");
    assert_eq!(error.code.as_ref(), "windows_ml_dtype_unsupported");
    assert_eq!(error.stage, InitializationStage::BufferPrepare);
}

#[test]
fn machine_report_requires_official_windows_ml_identity_and_all_runtime_stages() {
    for target in ["win-x64", "win-arm64"] {
        let report = runtime_report(target);
        let report = WindowsMlMachineReport::parse_and_validate_runtime_verified(
            &report.to_string(),
            target,
        )
        .expect("official Windows ML report");
        assert_eq!(report.target(), Some(target));
        assert_eq!(report.provider_name(), Some("CPUExecutionProvider"));
        assert_eq!(report.accelerator_name(), Some("CPU"));
        assert!(report
            .runtime_version()
            .expect("runtime version")
            .contains("Microsoft.WindowsAppSDK.ML/2.1.74"));
    }

    let mut ordinary_ort = runtime_report("win-x64");
    ordinary_ort["runtime"]["sourcePackage"]["id"] = serde_json::json!("Microsoft.ML.OnnxRuntime");
    assert!(WindowsMlMachineReport::parse_and_validate_runtime_verified(
        &ordinary_ort.to_string(),
        "win-x64"
    )
    .is_err());

    let mut incomplete = runtime_report("win-x64");
    incomplete["catalogRegistrationCompleted"] = serde_json::json!(false);
    assert!(WindowsMlMachineReport::parse_and_validate_runtime_verified(
        &incomplete.to_string(),
        "win-x64"
    )
    .is_err());
    assert!(WindowsMlMachineReport::parse_and_validate_runtime_verified(
        &runtime_report("win-x64").to_string(),
        "win-arm64"
    )
    .is_err());
}

#[test]
fn checked_in_runner_uses_only_pinned_official_windows_ml_packages() {
    let project = include_str!("../tools/windows-ml-runner/WindowsMlRunner.csproj");
    let source = include_str!("../tools/windows-ml-runner/Program.cs");
    let lock = include_str!("../tools/windows-ml-runner/packages.lock.json");

    assert!(project.contains(
        "<PackageReference Include=\"Microsoft.WindowsAppSDK.ML\" Version=\"[2.1.74]\" />"
    ));
    assert!(project.contains(
        "<PackageReference Include=\"Microsoft.Windows.AI.MachineLearning\" Version=\"[2.1.74]\" />"
    ));
    assert!(project.contains("<WindowsAppSDKSelfContained>true</WindowsAppSDKSelfContained>"));
    assert!(!project.contains("Microsoft.WindowsAppSDK.Runtime"));
    assert!(!project.contains("PackageReference Include=\"Microsoft.ML.OnnxRuntime\""));
    for api in [
        "ExecutionProviderCatalog.GetDefault",
        "RegisterCertifiedAsync",
        "GetEpDevices",
        "AppendExecutionProvider",
        "GetEpDeviceForInputs",
        "EndProfiling",
    ] {
        assert!(source.contains(api), "missing official API guard: {api}");
    }
    assert!(source.contains("Linux ORT is not an accepted substitute"));
    assert!(source.contains("Microsoft.Windows.AI.MachineLearning.Projection"));
    assert!(source.contains("performanceRuns must be between 0 and 100"));
    assert!(source.contains("PerformanceWarmupRuns = 5"));
    assert!(lock.contains("net8.0-windows10.0.17763/win-x64"));
    assert!(lock.contains("net8.0-windows10.0.17763/win-arm64"));
    assert!(!lock.contains("Microsoft.WindowsAppSDK.Runtime"));
}

#[cfg(not(target_os = "windows"))]
#[test]
fn linux_initialization_is_one_structured_fallback_without_runner_execution() {
    let factory = WindowsMlAdapterFactory::new(
        manifest("onnx"),
        WindowsMlAdapterConfig::new(
            "this-path-must-not-be-read.onnx",
            WindowsMlRunnerCommand::new("this-runner-must-not-execute"),
            Duration::from_secs(15),
        ),
    );
    assert!(matches!(
        factory.selector().select_once(&request()),
        AdapterSelection::Ready { selected }
            if selected.backend_kind == BackendKind::WindowsMl
                && selected.artifact_id == "windows-ml-onnx-fp32"
    ));
    let runtime = RuntimeLifecycle::new();
    let first = runtime
        .initialize_native(&request(), &factory)
        .expect("fallback is an outcome");
    let second = runtime
        .initialize_native(&request(), &factory)
        .expect("fallback is stable");
    assert_eq!(first, second);
    assert!(matches!(
        first,
        rimeflow_onnx_base::InitOutcome::UseWebFallback { failure }
            if failure.code.as_ref() == "native_runtime_unavailable"
                && failure.stage == InitializationStage::RuntimeLoad
                && failure.attempted_backend == Some(BackendKind::WindowsMl)
    ));
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(runtime.web_fallback_count(), 1);
}

#[test]
fn no_runner_conformance_reports_keep_x64_and_arm64_independently_blocked() {
    for arch in ["x86_64", "aarch64"] {
        let target = Platform::new("windows", arch);
        let report = AdapterConformanceReport {
            schema_version: ADAPTER_CONFORMANCE_SCHEMA_VERSION,
            case: AdapterConformanceCase {
                id: format!("windows-ml-{arch}"),
                model_id: "rimeflow-yolov8n".to_owned(),
                model_version: "yolov8n-onnx-20260707".to_owned(),
                target: target.clone(),
                adapter: BackendKind::WindowsMl,
                artifact_id: "windows-ml-onnx-fp32".to_owned(),
                artifact_sha256: MODEL_SHA256.to_owned(),
                manifest_sha256: MANIFEST_SHA256.to_owned(),
                native_initialization_timeout_ms: 15_000,
            },
            runner: ConformanceRunner {
                kind: ConformanceEvidenceKind::Unavailable,
                target,
                runner_id: None,
            },
            selection: AdapterSelection::UseWebFallback {
                failure: InitFailure::new(
                    "native_runtime_unavailable",
                    InitializationStage::RuntimeLoad,
                    "a real Windows target runner is not recorded",
                ),
            },
            checks: AdapterConformanceCheckKind::ALL
                .iter()
                .copied()
                .map(|kind| AdapterConformanceCheck {
                    kind,
                    status: match kind {
                        AdapterConformanceCheckKind::ManifestIo
                        | AdapterConformanceCheckKind::FaultInjection
                        | AdapterConformanceCheckKind::Diagnostics => {
                            AdapterConformanceStatus::BuildVerified
                        }
                        _ => AdapterConformanceStatus::Blocked,
                    },
                    detail: match kind {
                        AdapterConformanceCheckKind::ManifestIo => {
                            "Rust manifest and role mapping is statically verified"
                        }
                        AdapterConformanceCheckKind::FaultInjection => {
                            "runner identity and lifecycle rejection guards are statically verified"
                        }
                        AdapterConformanceCheckKind::Diagnostics => {
                            "machine-report identity validation is statically verified"
                        }
                        _ => "no real target runner result is recorded",
                    }
                    .to_owned(),
                    evidence_path: matches!(
                        kind,
                        AdapterConformanceCheckKind::ManifestIo
                            | AdapterConformanceCheckKind::FaultInjection
                            | AdapterConformanceCheckKind::Diagnostics
                    )
                    .then(|| "tests/windows_ml_adapter.rs".to_owned()),
                })
                .collect(),
        };
        report.validate().expect("honest blocked report");
        assert_eq!(report.overall_status(), AdapterConformanceStatus::Blocked);
    }
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("windows", "x86_64"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-onnx-20260707".to_owned(),
        artifact_id: "windows-ml-onnx-fp32".to_owned(),
        artifact_sha256: MODEL_SHA256.to_owned(),
    }
}

fn manifest(format: &str) -> ModelManifest {
    ModelManifest::parse_and_validate(&manifest_value(format).to_string()).expect("valid manifest")
}

fn manifest_value(format: &str) -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1,
        "model": {
            "id": "rimeflow-yolov8n",
            "version": "yolov8n-onnx-20260707"
        },
        "tensors": {
            "inputs": [{
                "role": "image",
                "name": "images",
                "shape": [1, 3, 640, 640],
                "layout": "NCHW",
                "dtype": "f32"
            }],
            "outputs": [{
                "role": "detections",
                "name": "output0",
                "shape": [1, 84, 8400],
                "layout": "NCHW",
                "dtype": "f32"
            }]
        },
        "artifacts": [{
            "id": "windows-ml-onnx-fp32",
            "format": format,
            "targets": [
                { "os": "windows", "arch": "x86_64" },
                { "os": "windows", "arch": "aarch64" }
            ],
            "path": "models/yolov8n.onnx",
            "sha256": MODEL_SHA256,
            "inputs": ["image"],
            "outputs": ["detections"]
        }]
    })
}

fn runtime_report(target: &str) -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1,
        "state": "runtime-verified",
        "target": target,
        "runtimeExecuted": true,
        "failureStage": "artifact-publication",
        "windowsMlApiCalled": true,
        "catalogRegistrationAttempted": true,
        "catalogRegistrationCompleted": true,
        "sessionCreated": true,
        "inferenceExecuted": true,
        "runtimeIntrospectionComplete": true,
        "outputPublished": true,
        "runtime": {
            "sourcePackage": {
                "id": "Microsoft.WindowsAppSDK.ML",
                "version": "2.1.74"
            },
            "runtimePackage": {
                "id": "Microsoft.Windows.AI.MachineLearning",
                "version": "2.1.74"
            },
            "ortVersion": "1.24.6"
        },
        "execution": {
            "selectedDevice": {
                "epName": "CPUExecutionProvider",
                "hardware": { "type": "CPU" }
            }
        }
    })
}
