use std::sync::atomic::{AtomicUsize, Ordering};

use rimeflow_onnx_base::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, BackendFactory,
    BackendInitRequest, BackendInstance, BackendKind, CapabilityStatus, ConformanceEvidenceKind,
    ConformanceRunner, DType, InferenceError, InitializationStage, ModelInput, ModelManifest,
    NativeAdapterCapability, OneShotNativeAdapterFactory, Platform, PlatformAdapterFactory,
    RawModelOutput, RawTensor, RuntimeBackend, RuntimeLifecycle, SelectedNativeAdapter, TensorData,
    ADAPTER_CONFORMANCE_SCHEMA_V1, ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};

const MODEL_SHA256: &str = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad";
const MANIFEST_SHA256: &str = "6f411aedec1550f3306459468dc3b4a0a4bc2558f5233f5f25404f1ac50e9c26";

struct TestBackend {
    fail_inference: bool,
}

impl RuntimeBackend for TestBackend {
    fn infer(&mut self, _input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        if self.fail_inference {
            return Err(InferenceError::new(
                "inference_failed",
                "deterministic post-readiness failure",
            ));
        }
        Ok(RawModelOutput {
            tensors: vec![RawTensor {
                role: "detections".to_owned(),
                shape: vec![1],
                data: TensorData::F32(vec![0.0]),
            }],
        })
    }
}

#[test]
fn platform_factory_selects_from_manifest_and_capability_exactly_once() {
    let selector = PlatformAdapterFactory::new(sample_manifest(), vec![linux_capability()]);
    let request = request();
    let first = selector.select_once(&request);
    let changed_request = BackendInitRequest {
        target: Platform::new("windows", "x86_64"),
        artifact_id: "different-artifact".to_owned(),
        ..request.clone()
    };
    let second = selector.select_once(&changed_request);

    assert_eq!(first, second, "the first selection is terminal");
    assert_eq!(selector.selection_evaluation_count(), 1);
    assert!(matches!(
        first,
        AdapterSelection::Ready { ref selected }
            if selected.backend_kind == BackendKind::OpenVino
                && selected.platform == Platform::new("linux", "x86_64")
                && selected.artifact_id == "linux-openvino-onnx-fp32"
                && selected.artifact_sha256 == MODEL_SHA256
    ));
}

#[test]
fn every_unready_native_prerequisite_returns_structured_web_fallback() {
    let cases = [
        (
            "artifact",
            InitializationStage::ArtifactIntegrity,
            "adapter_or_artifact_unavailable",
        ),
        (
            "runtime",
            InitializationStage::RuntimeLoad,
            "native_runtime_unavailable",
        ),
        (
            "device",
            InitializationStage::DeviceCreate,
            "native_device_unavailable",
        ),
        (
            "smoke",
            InitializationStage::SmokeInference,
            "native_smoke_failed",
        ),
    ];

    for (unready, expected_stage, expected_code) in cases {
        let mut capability = linux_capability();
        let blocked = CapabilityStatus::blocked(format!("injected {unready} blocker"));
        match unready {
            "artifact" => capability.artifact = blocked,
            "runtime" => capability.runtime = blocked,
            "device" => capability.device = blocked,
            "smoke" => capability.smoke = blocked,
            _ => unreachable!(),
        }
        let selector = PlatformAdapterFactory::new(sample_manifest(), vec![capability]);
        assert!(matches!(
            selector.select_once(&request()),
            AdapterSelection::UseWebFallback { failure }
                if failure.stage == expected_stage && failure.code.as_ref() == expected_code
        ));
        assert_eq!(selector.selection_evaluation_count(), 1);
    }
}

#[test]
fn ready_runtime_returns_inference_errors_without_rebuild_or_switch() {
    let builds = AtomicUsize::new(0);
    let selector = PlatformAdapterFactory::new(sample_manifest(), vec![linux_capability()]);
    let factory = OneShotNativeAdapterFactory::new(
        selector,
        |request: &BackendInitRequest, selected: &SelectedNativeAdapter| {
            builds.fetch_add(1, Ordering::SeqCst);
            Ok(BackendInstance {
                backend: TestBackend {
                    fail_inference: true,
                },
                resolved: selected.resolved_backend(request, 3),
            })
        },
    );
    let lifecycle = RuntimeLifecycle::new();

    let first = lifecycle
        .initialize_native(&request(), &factory)
        .expect("first selection succeeds");
    let second = lifecycle
        .initialize_native(&request(), &factory)
        .expect("repeated initialization returns the fixed selection");
    assert_eq!(first, second);
    for _ in 0..3 {
        let error = lifecycle
            .infer(ModelInput::Tensor {
                role: "image".to_owned(),
                shape: vec![1],
                dtype: DType::F32,
                bytes: 0.0f32.to_le_bytes().to_vec(),
            })
            .expect_err("inference failure is returned to the caller");
        assert_eq!(error.code, "inference_failed");
    }
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
    assert_eq!(lifecycle.published_instance_count(), 1);
    assert_eq!(lifecycle.web_fallback_count(), 0);
    assert_eq!(
        lifecycle
            .diagnostics()
            .expect("ready diagnostics")
            .backend_kind,
        BackendKind::OpenVino
    );
}

#[test]
fn direct_factory_rejects_a_second_backend_build_attempt() {
    let selector = PlatformAdapterFactory::new(sample_manifest(), vec![linux_capability()]);
    let factory = OneShotNativeAdapterFactory::new(
        selector,
        |request: &BackendInitRequest, selected: &SelectedNativeAdapter| {
            Ok(BackendInstance {
                backend: TestBackend {
                    fail_inference: false,
                },
                resolved: selected.resolved_backend(request, 1),
            })
        },
    );
    drop(factory.create(&request()).expect("first direct build"));
    let error = match factory.create(&request()) {
        Ok(_) => panic!("second direct build is forbidden"),
        Err(error) => error,
    };
    assert_eq!(error.code.as_ref(), "native_factory_rebuild_forbidden");
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
}

#[test]
fn conformance_report_requires_complete_honest_check_statuses() {
    let selection = PlatformAdapterFactory::new(sample_manifest(), vec![linux_capability()])
        .select_once(&request());
    let report = AdapterConformanceReport {
        schema_version: ADAPTER_CONFORMANCE_SCHEMA_VERSION,
        case: linux_case(),
        runner: ConformanceRunner {
            kind: ConformanceEvidenceKind::RealTarget,
            target: Platform::new("linux", "x86_64"),
            runner_id: Some("local-linux-x86_64".to_owned()),
        },
        selection,
        checks: AdapterConformanceCheckKind::ALL
            .iter()
            .copied()
            .map(|kind| AdapterConformanceCheck {
                kind,
                status: match kind {
                    AdapterConformanceCheckKind::Performance => {
                        AdapterConformanceStatus::BuildVerified
                    }
                    AdapterConformanceCheckKind::PackageLoad => AdapterConformanceStatus::Blocked,
                    _ => AdapterConformanceStatus::Passed,
                },
                detail: format!("{kind:?} has an explicit result"),
                evidence_path: (kind != AdapterConformanceCheckKind::PackageLoad)
                    .then(|| "tests/platform_factory.rs".to_owned()),
            })
            .collect(),
    };
    report.validate().expect("complete real-target report");
    assert_eq!(report.overall_status(), AdapterConformanceStatus::Blocked);

    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    assert_eq!(
        AdapterConformanceReport::parse_and_validate(&json).expect("round trip"),
        report
    );
    let schema: serde_json::Value =
        serde_json::from_str(ADAPTER_CONFORMANCE_SCHEMA_V1).expect("checked-in JSON schema");
    assert_eq!(schema["properties"]["schemaVersion"]["const"], 1);

    let mut dishonest = report;
    dishonest.runner = ConformanceRunner {
        kind: ConformanceEvidenceKind::Unavailable,
        target: Platform::new("linux", "x86_64"),
        runner_id: None,
    };
    assert!(
        dishonest.validate().is_err(),
        "Ready cannot claim no runner"
    );
}

fn linux_case() -> AdapterConformanceCase {
    AdapterConformanceCase {
        id: "linux-x86_64-openvino".to_owned(),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-onnx-20260707".to_owned(),
        target: Platform::new("linux", "x86_64"),
        adapter: BackendKind::OpenVino,
        artifact_id: "linux-openvino-onnx-fp32".to_owned(),
        artifact_sha256: MODEL_SHA256.to_owned(),
        manifest_sha256: MANIFEST_SHA256.to_owned(),
        native_initialization_timeout_ms: 15_000,
    }
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("linux", "x86_64"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-onnx-20260707".to_owned(),
        artifact_id: "linux-openvino-onnx-fp32".to_owned(),
        artifact_sha256: MODEL_SHA256.to_owned(),
    }
}

fn linux_capability() -> NativeAdapterCapability {
    let mut capability = NativeAdapterCapability::ready(
        BackendKind::OpenVino,
        Platform::new("linux", "x86_64"),
        vec![rimeflow_onnx_base::ArtifactFormat::Onnx],
    );
    capability.configured_provider = Some("CPU".to_owned());
    capability.runtime_version = Some("ort-2.0.0-rc.12".to_owned());
    capability
}

fn sample_manifest() -> ModelManifest {
    ModelManifest::parse_and_validate(include_str!(
        "fixtures/conformance/linux-openvino-manifest.json"
    ))
    .expect("sample manifest")
}
