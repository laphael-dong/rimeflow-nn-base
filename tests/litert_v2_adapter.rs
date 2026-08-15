use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rimeflow_onnx_base::backend::litert_v2::{
    quantize_f32, LiteRtCompiledRuntime, LiteRtRuntimeError, LiteRtTensorBinding,
    LiteRtTensorDescriptor, LiteRtV2Availability, LiteRtV2Backend, LiteRtV2BootstrapError,
    VerifiedLiteRtArtifact, LITERT_RUNTIME_VERSION,
};
use rimeflow_onnx_base::manifest::{sha256_hex, ModelIdentity, Quantization, TensorGroups};
use rimeflow_onnx_base::{
    AdapterConformanceReport, AdapterSelection, Artifact, ArtifactFormat, ArtifactTarget,
    BackendInitRequest, BackendKind, CapabilityStatus, DType, ExecutionPlan, InitializationStage,
    Layout, ModelInput, ModelManifest, Platform, PlatformAdapterFactory, ResolvedBackend,
    RuntimeBackend, TensorData, TensorSpec,
};

const ARTIFACT_BYTES: &[u8] = b"minimal-tflite-fixture";

struct FakeCompiledModel {
    inputs: Vec<LiteRtTensorDescriptor>,
    outputs: Vec<LiteRtTensorDescriptor>,
    output_data: Vec<TensorData>,
    runs: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    last_input: Arc<Mutex<Option<TensorData>>>,
    fail: bool,
}

impl LiteRtCompiledRuntime for FakeCompiledModel {
    fn input_descriptors(&self) -> &[LiteRtTensorDescriptor] {
        &self.inputs
    }

    fn output_descriptors(&self) -> &[LiteRtTensorDescriptor] {
        &self.outputs
    }

    fn run(&mut self, inputs: &[TensorData]) -> Result<Vec<TensorData>, LiteRtRuntimeError> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        *self.last_input.lock().expect("last input lock") = inputs.first().cloned();
        if self.fail {
            return Err(LiteRtRuntimeError::new(
                "injected_smoke_failure",
                "deterministic fake failure",
            ));
        }
        Ok(self.output_data.clone())
    }
}

impl Drop for FakeCompiledModel {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn nhwc_u8_role_mapping_quantizes_and_reuses_one_compiled_model() {
    let manifest = manifest(DType::U8, 0);
    let request = request(&manifest);
    let verified = VerifiedLiteRtArtifact::verify(manifest, &request, ARTIFACT_BYTES)
        .expect("verified TFLite fixture");
    let runs = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let last_input = Arc::new(Mutex::new(None));
    let runtime = fake_runtime(
        DType::U8,
        runs.clone(),
        drops.clone(),
        last_input.clone(),
        false,
    );
    let mut backend =
        LiteRtV2Backend::from_verified_artifact(verified, runtime, resolved(&request))
            .expect("smoke-verified adapter");

    assert_eq!(runs.load(Ordering::SeqCst), 1, "initial smoke run");
    assert_eq!(backend.diagnostics().io_plan.inputs[0].role, "image");
    assert_eq!(
        backend.diagnostics().io_plan.inputs[0].runtime_name,
        "serving_default_image:0"
    );
    assert_eq!(backend.diagnostics().io_plan.inputs[0].layout, Layout::Nhwc);
    assert_eq!(
        backend.diagnostics().runtime_version,
        LITERT_RUNTIME_VERSION
    );

    let input_values = [0.0f32, 0.5, 1.0, -1.0, 2.0, 0.25];
    let output = backend
        .infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1, 1, 2, 3],
            dtype: DType::F32,
            bytes: input_values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        })
        .expect("manifest-driven quantized inference");
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert_eq!(output.tensors[0].role, "detections");
    assert_eq!(output.tensors[0].shape, vec![1, 1, 1, 2]);
    assert_eq!(
        *last_input.lock().expect("last input lock"),
        Some(TensorData::U8(vec![0, 128, 255, 0, 255, 64]))
    );

    drop(backend);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn i8_quantization_uses_manifest_scale_zero_point_and_clamps() {
    let binding = LiteRtTensorBinding {
        role: "image".to_owned(),
        runtime_name: "image".to_owned(),
        runtime_index: 0,
        shape: vec![1, 1, 1, 5],
        layout: Layout::Nhwc,
        dtype: DType::I8,
        quantization_scale: Some(0.5),
        quantization_zero_point: Some(-3),
    };
    assert_eq!(
        quantize_f32(&binding, &[-100.0, -0.5, 0.0, 1.0, 100.0]).expect("i8 quantization"),
        TensorData::I8(vec![-128, -4, -3, -1, 127])
    );
}

#[test]
fn artifact_io_and_smoke_failures_keep_structured_stages() {
    let manifest = manifest(DType::U8, 0);
    let request = request(&manifest);
    let artifact_error = VerifiedLiteRtArtifact::verify(manifest.clone(), &request, b"tampered")
        .expect_err("digest mismatch");
    assert_eq!(artifact_error.stage, InitializationStage::ArtifactIntegrity);
    assert_eq!(
        artifact_error.code.as_ref(),
        "adapter_or_artifact_unavailable"
    );

    let verified = VerifiedLiteRtArtifact::verify(manifest.clone(), &request, ARTIFACT_BYTES)
        .expect("verified fixture");
    let io_error = match LiteRtV2Backend::from_verified_artifact(
        verified,
        FakeCompiledModel {
            inputs: vec![LiteRtTensorDescriptor {
                name: "wrong_runtime_name".to_owned(),
                index: 0,
                shape: vec![1, 1, 2, 3],
                dtype: DType::U8,
            }],
            outputs: vec![output_descriptor()],
            output_data: vec![TensorData::F32(vec![0.0, 1.0])],
            runs: Arc::new(AtomicUsize::new(0)),
            drops: Arc::new(AtomicUsize::new(0)),
            last_input: Arc::new(Mutex::new(None)),
            fail: false,
        },
        resolved(&request),
    ) {
        Ok(_) => panic!("runtime name mismatch must fail"),
        Err(error) => error,
    };
    assert_eq!(io_error.stage, InitializationStage::IoDiscovery);
    assert_eq!(io_error.code.as_ref(), "litert_io_contract_mismatch");

    let verified = VerifiedLiteRtArtifact::verify(manifest, &request, ARTIFACT_BYTES)
        .expect("verified fixture");
    let smoke_error = match LiteRtV2Backend::from_verified_artifact(
        verified,
        fake_runtime(
            DType::U8,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(Mutex::new(None)),
            true,
        ),
        resolved(&request),
    ) {
        Ok(_) => panic!("smoke failure must fail initialization"),
        Err(error) => error,
    };
    assert_eq!(smoke_error.stage, InitializationStage::SmokeInference);
    assert_eq!(smoke_error.code.as_ref(), "native_smoke_failed");
}

#[test]
fn litert_capability_maps_artifact_runtime_device_and_smoke_fallbacks() {
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
    for (blocked, expected_stage, expected_code) in cases {
        let blocked_status = CapabilityStatus::blocked(format!("injected {blocked} blocker"));
        let availability = LiteRtV2Availability {
            artifact: if blocked == "artifact" {
                blocked_status.clone()
            } else {
                CapabilityStatus::Ready
            },
            runtime: if blocked == "runtime" {
                blocked_status.clone()
            } else {
                CapabilityStatus::Ready
            },
            device: if blocked == "device" {
                blocked_status.clone()
            } else {
                CapabilityStatus::Ready
            },
            smoke: if blocked == "smoke" {
                blocked_status
            } else {
                CapabilityStatus::Ready
            },
            accelerator: Some("CPU".to_owned()),
        };
        let manifest = manifest(DType::U8, 0);
        let request = request(&manifest);
        let selection = PlatformAdapterFactory::new(
            manifest,
            vec![availability.into_native_capability(request.target.clone())],
        )
        .select_once(&request);
        assert!(matches!(
            selection,
            AdapterSelection::UseWebFallback { failure }
                if failure.stage == expected_stage && failure.code.as_ref() == expected_code
        ));
    }

    assert_eq!(
        LiteRtV2BootstrapError::runtime("missing package")
            .into_init_failure(&request(&manifest(DType::U8, 0)))
            .stage,
        InitializationStage::RuntimeLoad
    );
    assert_eq!(
        LiteRtV2BootstrapError::device("missing device")
            .into_init_failure(&request(&manifest(DType::U8, 0)))
            .stage,
        InitializationStage::DeviceCreate
    );
}

#[test]
fn build_only_conformance_report_is_honest_about_missing_android_runner() {
    let report = AdapterConformanceReport::parse_and_validate(include_str!(
        "../reports/os6-base-litert-v2-conformance.json"
    ))
    .expect("checked-in conformance report");
    assert_eq!(report.case.adapter, BackendKind::LiteRtV2);
    assert!(matches!(
        report.selection,
        AdapterSelection::UseWebFallback { ref failure }
            if failure.stage == InitializationStage::RuntimeLoad
    ));
    assert_ne!(
        report.overall_status(),
        rimeflow_onnx_base::AdapterConformanceStatus::Passed
    );
}

fn fake_runtime(
    input_dtype: DType,
    runs: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
    last_input: Arc<Mutex<Option<TensorData>>>,
    fail: bool,
) -> FakeCompiledModel {
    FakeCompiledModel {
        inputs: vec![LiteRtTensorDescriptor {
            name: "serving_default_image:0".to_owned(),
            index: 0,
            shape: vec![1, 1, 2, 3],
            dtype: input_dtype,
        }],
        outputs: vec![output_descriptor()],
        output_data: vec![TensorData::F32(vec![0.0, 1.0])],
        runs,
        drops,
        last_input,
        fail,
    }
}

fn output_descriptor() -> LiteRtTensorDescriptor {
    LiteRtTensorDescriptor {
        name: "StatefulPartitionedCall:0".to_owned(),
        index: 0,
        shape: vec![1, 1, 1, 2],
        dtype: DType::F32,
    }
}

fn manifest(input_dtype: DType, zero_point: i64) -> ModelManifest {
    let digest = sha256_hex(ARTIFACT_BYTES);
    ModelManifest {
        schema_version: 1,
        model: ModelIdentity {
            id: "rimeflow-yolov8n".to_owned(),
            version: "litert-test-v1".to_owned(),
        },
        tensors: TensorGroups {
            inputs: vec![TensorSpec {
                role: "image".to_owned(),
                name: Some("serving_default_image:0".to_owned()),
                index: Some(0),
                shape: vec![1, 1, 2, 3],
                layout: Layout::Nhwc,
                dtype: input_dtype,
                quantization: Some(Quantization {
                    scale: 1.0 / 255.0,
                    zero_point: Some(zero_point),
                }),
            }],
            outputs: vec![TensorSpec {
                role: "detections".to_owned(),
                name: Some("StatefulPartitionedCall:0".to_owned()),
                index: Some(0),
                shape: vec![1, 1, 1, 2],
                layout: Layout::Nchw,
                dtype: DType::F32,
                quantization: None,
            }],
        },
        artifacts: vec![Artifact {
            id: "yolov8n-tflite-quantized".to_owned(),
            format: ArtifactFormat::Tflite,
            targets: vec![ArtifactTarget {
                os: "android".to_owned(),
                arch: "arm64".to_owned(),
            }],
            path: "models/yolov8n.tflite".to_owned(),
            sha256: digest,
            converter: None,
            inputs: vec!["image".to_owned()],
            outputs: vec!["detections".to_owned()],
        }],
    }
}

fn request(manifest: &ModelManifest) -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("android", "arm64"),
        model_id: manifest.model.id.clone(),
        model_version: manifest.model.version.clone(),
        artifact_id: manifest.artifacts[0].id.clone(),
        artifact_sha256: manifest.artifacts[0].sha256.clone(),
    }
}

fn resolved(request: &BackendInitRequest) -> ResolvedBackend {
    ResolvedBackend {
        backend_kind: BackendKind::LiteRtV2,
        platform: request.target.clone(),
        configured_provider: Some("LiteRT CompiledModel".to_owned()),
        accelerator: Some("CPU".to_owned()),
        execution_plan: ExecutionPlan::Unknown,
        model_version: request.model_version.clone(),
        artifact_id: request.artifact_id.clone(),
        artifact_sha256: request.artifact_sha256.clone(),
        initialization_ms: 1,
        runtime_version: Some(LITERT_RUNTIME_VERSION.to_owned()),
    }
}
