#![cfg(all(not(target_arch = "wasm32"), feature = "native"))]

use std::fs;
use std::path::PathBuf;

use rimeflow_onnx_base::manifest::sha256_hex;
use rimeflow_onnx_base::{
    ExecutionPlan, LegacyInferError, LegacyOrtBackend, LegacyOrtMetadata, NativeOrtBackend,
    Platform, TensorData,
};

const LOCKED_MODEL_SHA256: &str =
    "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad";
const LOCKED_MODEL_BYTES: usize = 12_851_098;
const OUTPUT_ELEMENTS: usize = 84 * 8400;

#[test]
fn legacy_module_constructor_and_root_reexports_remain_available() {
    let _module_constructor: fn(
        &[u8],
        &wgpu::Device,
        &wgpu::Queue,
        u32,
    ) -> Result<
        rimeflow_onnx_base::native_ort::NativeOrtBackend,
        LegacyInferError,
    > = rimeflow_onnx_base::native_ort::NativeOrtBackend::new;
    let _root_constructor: fn(
        &[u8],
        &wgpu::Device,
        &wgpu::Queue,
        u32,
    ) -> Result<NativeOrtBackend, LegacyInferError> = NativeOrtBackend::new;
}

#[test]
#[ignore = "requires the locked validation YOLOv8n model via RIMEFLOW_YOLOV8N_MODEL"]
fn real_yolov8n_legacy_adapter_matches_native_ort_output() {
    let model_path = PathBuf::from(
        std::env::var_os("RIMEFLOW_YOLOV8N_MODEL")
            .expect("RIMEFLOW_YOLOV8N_MODEL must name the locked model"),
    );
    let model_bytes = fs::read(&model_path).expect("read locked model");
    assert_eq!(model_bytes.len(), LOCKED_MODEL_BYTES);
    assert_eq!(sha256_hex(&model_bytes), LOCKED_MODEL_SHA256);

    let mut native = NativeOrtBackend::from_model_bytes(&model_bytes, 640)
        .expect("legacy NativeOrtBackend initializes");
    let mut adapter = LegacyOrtBackend::from_model_bytes(
        &model_bytes,
        640,
        LegacyOrtMetadata {
            platform: Platform::new("linux", "x86_64"),
            model_version: "8.0.0".to_owned(),
            artifact_id: "yolov8n-onnx-fp32".to_owned(),
            artifact_sha256: LOCKED_MODEL_SHA256.to_owned(),
            output_role: "detections".to_owned(),
            output_shape: vec![1, 84, 8400],
            runtime_version: Some("ort-2.0.0-rc.12".to_owned()),
        },
    )
    .expect("LegacyOrtBackend initializes");
    let input = vec![0.0f32; 3 * 640 * 640];
    let native_output = native
        .infer_from_host_slice(&input)
        .expect("Native ORT smoke inference");
    let adapter_output = adapter
        .infer_from_host_slice(&input)
        .expect("adapter smoke inference");
    let TensorData::F32(adapter_values) = &adapter_output.tensors[0].data else {
        panic!("Legacy adapter output must be f32");
    };

    assert_eq!(native_output.len(), OUTPUT_ELEMENTS);
    assert_eq!(adapter_values.len(), OUTPUT_ELEMENTS);
    let max_difference = native_output
        .iter()
        .zip(adapter_values)
        .map(|(native, adapter)| (native - adapter).abs())
        .fold(0.0f32, f32::max);
    assert!(max_difference <= 1e-6, "max difference {max_difference}");
    assert!(adapter_values.iter().all(|value| value.is_finite()));

    let resolved = adapter.resolved_backend();
    assert_eq!(resolved.configured_provider.as_deref(), Some("CPU"));
    assert_eq!(resolved.accelerator, None);
    assert_eq!(resolved.execution_plan, ExecutionPlan::Unknown);
}
