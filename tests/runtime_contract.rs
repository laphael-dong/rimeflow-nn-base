use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use rimeflow_onnx_base::{
    BackendFactory, BackendInitRequest, BackendInstance, BackendKind, DType, ExecutionPlan,
    InferenceError, InitFailure, InitOutcome, InitializationStage, LifecycleSnapshot, ModelInput,
    Platform, RawModelOutput, RawTensor, ResolvedBackend, RuntimeBackend, RuntimeLifecycle,
    TensorData, WebInitOutcome,
};

struct ResourceCounters {
    live: AtomicUsize,
    drops: AtomicUsize,
}

struct TestBackend {
    counters: Arc<ResourceCounters>,
    inference_fails: bool,
}

impl Drop for TestBackend {
    fn drop(&mut self) {
        self.counters.live.fetch_sub(1, Ordering::SeqCst);
        self.counters.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl RuntimeBackend for TestBackend {
    fn infer(&mut self, _input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        if self.inference_fails {
            return Err(InferenceError::new(
                "inference_failed",
                "deterministic post-readiness fault",
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

struct CountingFactory {
    creates: AtomicUsize,
    counters: Arc<ResourceCounters>,
    failure: Option<InitFailure>,
    inference_fails: bool,
    delay: Duration,
    resolved_kind: BackendKind,
}

impl CountingFactory {
    fn succeeds(inference_fails: bool) -> Self {
        Self::new(None, inference_fails, BackendKind::LegacyOrt)
    }

    fn fails(failure: InitFailure) -> Self {
        Self::new(Some(failure), false, BackendKind::LegacyOrt)
    }

    fn new(
        failure: Option<InitFailure>,
        inference_fails: bool,
        resolved_kind: BackendKind,
    ) -> Self {
        Self {
            creates: AtomicUsize::new(0),
            counters: Arc::new(ResourceCounters {
                live: AtomicUsize::new(0),
                drops: AtomicUsize::new(0),
            }),
            failure,
            inference_fails,
            delay: Duration::from_millis(15),
            resolved_kind,
        }
    }
}

impl BackendFactory<TestBackend> for CountingFactory {
    fn create(
        &self,
        request: &BackendInitRequest,
    ) -> Result<BackendInstance<TestBackend>, InitFailure> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        self.counters.live.fetch_add(1, Ordering::SeqCst);
        let backend = TestBackend {
            counters: Arc::clone(&self.counters),
            inference_fails: self.inference_fails,
        };
        thread::sleep(self.delay);
        if let Some(failure) = &self.failure {
            drop(backend);
            return Err(failure.clone());
        }
        Ok(BackendInstance {
            backend,
            resolved: resolved(request, self.resolved_kind),
        })
    }
}

#[test]
fn concurrent_initialization_publishes_once_and_release_is_terminal() {
    let runtime = Arc::new(RuntimeLifecycle::new());
    let factory = Arc::new(CountingFactory::succeeds(false));
    let request = Arc::new(request());
    let workers: Vec<_> = (0..12)
        .map(|_| {
            let runtime = Arc::clone(&runtime);
            let factory = Arc::clone(&factory);
            let request = Arc::clone(&request);
            thread::spawn(move || runtime.initialize_native(&request, &*factory).unwrap())
        })
        .collect();
    let outcomes: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("initializer thread"))
        .collect();

    assert!(outcomes.iter().all(|outcome| outcome == &outcomes[0]));
    assert!(matches!(outcomes[0], InitOutcome::Ready { .. }));
    assert_eq!(factory.creates.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.published_instance_count(), 1);
    assert_eq!(factory.counters.live.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.snapshot(), LifecycleSnapshot::ReadyNative);

    runtime.release().expect("release succeeds");
    assert_eq!(factory.counters.live.load(Ordering::SeqCst), 0);
    assert_eq!(factory.counters.drops.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.snapshot(), LifecycleSnapshot::Released);
    let error = runtime
        .initialize_native(&request, &*factory)
        .expect_err("released runtime cannot rebuild");
    assert_eq!(error.code, "runtime_released");
    assert_eq!(factory.creates.load(Ordering::SeqCst), 1);
    let error = runtime
        .infer(tensor_input())
        .expect_err("released runtime cannot infer");
    assert_eq!(error.code, "runtime_released");
}

#[test]
fn every_stage_failure_is_atomic_and_returns_one_fallback() {
    let stages = [
        InitializationStage::ManifestParse,
        InitializationStage::ArtifactIntegrity,
        InitializationStage::RuntimeLoad,
        InitializationStage::DeviceCreate,
        InitializationStage::ModelCompile,
        InitializationStage::IoDiscovery,
        InitializationStage::BufferPrepare,
        InitializationStage::SmokeInference,
    ];
    for stage in stages {
        let runtime = RuntimeLifecycle::new();
        let factory = CountingFactory::fails(InitFailure::new(
            "native_initialization_failed",
            stage,
            "injected stage failure",
        ));
        let first = runtime
            .initialize_native(&request(), &factory)
            .expect("fallback is a structured outcome");
        let second = runtime
            .initialize_native(&request(), &factory)
            .expect("fallback is stable");
        assert_eq!(first, second);
        assert!(matches!(
            first,
            InitOutcome::UseWebFallback {
                failure: InitFailure {
                    code,
                    stage: observed_stage,
                    ..
                }
            } if code.as_ref() == "native_initialization_failed" && observed_stage == stage
        ));
        assert_eq!(factory.creates.load(Ordering::SeqCst), 1);
        assert_eq!(factory.counters.live.load(Ordering::SeqCst), 0);
        assert_eq!(factory.counters.drops.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.published_instance_count(), 0);
        assert_eq!(runtime.web_fallback_count(), 1);
        assert_eq!(runtime.snapshot(), LifecycleSnapshot::UseWebFallback);
    }
}

#[test]
fn native_timeout_and_web_failure_are_stable_terminal_results() {
    let runtime = RuntimeLifecycle::new();
    let native_factory = CountingFactory::fails(InitFailure::new(
        "native_initialization_timeout",
        InitializationStage::RuntimeLoad,
        "injected Native timeout",
    ));
    let native = runtime
        .initialize_native(&request(), &native_factory)
        .expect("Native timeout selects fallback");
    assert!(matches!(
        native,
        InitOutcome::UseWebFallback { ref failure }
            if failure.code.as_ref() == "native_initialization_timeout"
    ));

    let web_factory = CountingFactory::new(
        Some(InitFailure::new(
            "web_initialization_timeout",
            InitializationStage::SmokeInference,
            "injected Web timeout",
        )),
        false,
        BackendKind::WebOnnx,
    );
    let first = runtime
        .initialize_web(&request(), &web_factory)
        .expect("Web timeout is a terminal result");
    let second = runtime
        .initialize_web(&request(), &web_factory)
        .expect("terminal result is stable");
    assert_eq!(first, second);
    assert!(matches!(
        first,
        WebInitOutcome::TerminalFailure { ref failure }
            if failure.code.as_ref() == "web_initialization_timeout"
    ));
    assert_eq!(web_factory.creates.load(Ordering::SeqCst), 1);
    assert_eq!(web_factory.counters.live.load(Ordering::SeqCst), 0);
    assert_eq!(runtime.snapshot(), LifecycleSnapshot::WebTerminalFailure);

    let retry_factory = CountingFactory::succeeds(false);
    let retry = runtime
        .initialize_native(&request(), &retry_factory)
        .expect("Native retry remains a fallback result");
    assert!(matches!(retry, InitOutcome::UseWebFallback { .. }));
    assert_eq!(retry_factory.creates.load(Ordering::SeqCst), 0);
}

#[test]
fn inference_error_never_switches_or_rebuilds_backend() {
    let runtime = RuntimeLifecycle::new();
    let factory = CountingFactory::succeeds(true);
    runtime
        .initialize_native(&request(), &factory)
        .expect("runtime initializes");
    let before = runtime.diagnostics().expect("diagnostics after readiness");
    for _ in 0..3 {
        let error = runtime
            .infer(tensor_input())
            .expect_err("inference fault is returned");
        assert_eq!(error.code, "inference_failed");
    }
    assert_eq!(runtime.diagnostics(), Some(before));
    assert_eq!(runtime.snapshot(), LifecycleSnapshot::ReadyNative);
    assert_eq!(factory.creates.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.published_instance_count(), 1);
    assert_eq!(runtime.web_fallback_count(), 0);
}

#[test]
fn resolved_backend_diagnostic_json_is_stable_and_honest() {
    let diagnostic = resolved(&request(), BackendKind::LegacyOrt);
    let value = serde_json::to_value(diagnostic).expect("serialize diagnostic");
    assert_eq!(
        value,
        serde_json::json!({
            "backendKind": "legacy_ort",
            "platform": { "os": "linux", "arch": "x86_64" },
            "configuredProvider": "CPU",
            "accelerator": null,
            "executionPlan": "unknown",
            "modelVersion": "8.0.0",
            "artifactId": "yolov8n-onnx-fp32",
            "artifactSha256": "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad",
            "initializationMs": 7,
            "runtimeVersion": "onnxruntime-locked"
        })
    );
}

#[test]
fn model_input_validates_exact_buffer_lengths() {
    let rgba = ModelInput::Rgba8 {
        width: 2,
        height: 2,
        bytes: vec![0; 15],
    };
    assert_eq!(
        rgba.validate().expect_err("short RGBA input").code,
        "invalid_rgba_shape"
    );
    let tensor = ModelInput::Tensor {
        role: "image".to_owned(),
        shape: vec![1, 3],
        dtype: DType::F32,
        bytes: vec![0; 8],
    };
    assert_eq!(
        tensor.validate().expect_err("short tensor input").code,
        "invalid_tensor_bytes"
    );
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("linux", "x86_64"),
        model_id: "yolov8n".to_owned(),
        model_version: "8.0.0".to_owned(),
        artifact_id: "yolov8n-onnx-fp32".to_owned(),
        artifact_sha256: "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad"
            .to_owned(),
    }
}

fn resolved(request: &BackendInitRequest, kind: BackendKind) -> ResolvedBackend {
    ResolvedBackend {
        backend_kind: kind,
        platform: request.target.clone(),
        configured_provider: Some("CPU".to_owned()),
        accelerator: None,
        execution_plan: ExecutionPlan::Unknown,
        model_version: request.model_version.clone(),
        artifact_id: request.artifact_id.clone(),
        artifact_sha256: request.artifact_sha256.clone(),
        initialization_ms: 7,
        runtime_version: Some("onnxruntime-locked".to_owned()),
    }
}

fn tensor_input() -> ModelInput {
    ModelInput::Tensor {
        role: "image".to_owned(),
        shape: vec![1],
        dtype: DType::F32,
        bytes: 0.0f32.to_le_bytes().to_vec(),
    }
}
