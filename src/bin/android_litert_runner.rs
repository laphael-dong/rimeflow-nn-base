#![cfg(target_os = "android")]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rimeflow_onnx_base::android_runner::{
    sha256_hex, AndroidBundleManifest, AndroidLoadedLibrary, AndroidRunnerDevice,
    AndroidRunnerOutput, AndroidRunnerPerformance, AndroidRunnerReport, AndroidRunnerReportState,
    ANDROID_RUNNER_REPORT_SCHEMA_VERSION,
};
use rimeflow_onnx_base::{
    AndroidLiteRtAccelerator, AndroidLiteRtV2Factory, BackendFactory, BackendInitRequest,
    ModelInput, ModelManifest, RuntimeBackend, TensorData, LITERT_RUNTIME_VERSION,
    LITERT_RUST_BINDING_VERSION,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bundle_root = required_option(&args, "--bundle").map(PathBuf::from);
    let report_path = required_option(&args, "--report").map(PathBuf::from);
    let exit = match (bundle_root, report_path) {
        (Ok(bundle_root), Ok(report_path)) => run(&bundle_root, &report_path),
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("{error}");
            2
        }
    };
    std::process::exit(exit);
}

fn run(bundle_root: &Path, report_path: &Path) -> i32 {
    let manifest_path = bundle_root.join("bundle-manifest.json");
    let manifest_bytes = match std::fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("无法读取 {}: {error}", manifest_path.display());
            return 2;
        }
    };
    let manifest = match std::str::from_utf8(&manifest_bytes)
        .map_err(|error| error.to_string())
        .and_then(AndroidBundleManifest::parse_and_validate)
        .and_then(|manifest| {
            manifest.validate_target_arch(compiled_bundle_arch()?)?;
            Ok(manifest)
        }) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("bundle_manifest_invalid: {error}");
            return 2;
        }
    };
    let mut report = base_report(&manifest, sha256_hex(&manifest_bytes));
    let result = execute(bundle_root, &manifest, &mut report);
    match result {
        Ok(()) => {
            report.state = AndroidRunnerReportState::RuntimeVerified;
            report.selection_code = "litert_v2_cpu_selected".to_owned();
        }
        Err((stage, code, message)) => {
            report.state = AndroidRunnerReportState::Failed;
            report.selection_code = code;
            report.failure_stage = Some(stage);
            report.failure_message = Some(message);
        }
    }
    if let Some(parent) = report_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_vec_pretty(&report)
        .map_err(|error| error.to_string())
        .and_then(|bytes| std::fs::write(report_path, bytes).map_err(|error| error.to_string()))
    {
        Ok(()) if report.state == AndroidRunnerReportState::RuntimeVerified => 0,
        Ok(()) => 1,
        Err(error) => {
            eprintln!("runner_report_write_failed: {error}");
            2
        }
    }
}

fn compiled_bundle_arch() -> Result<&'static str, String> {
    match std::env::consts::ARCH {
        "aarch64" => Ok("arm64"),
        "x86_64" => Ok("x86_64"),
        arch => Err(format!("runner 不支持编译架构 {arch}")),
    }
}

fn execute(
    root: &Path,
    manifest: &AndroidBundleManifest,
    report: &mut AndroidRunnerReport,
) -> Result<(), (String, String, String)> {
    manifest.verify_all_files(root).map_err(|message| {
        (
            "digest-verification".to_owned(),
            "bundle_digest_mismatch".to_owned(),
            message,
        )
    })?;
    let current_executable = std::env::current_exe()
        .and_then(std::fs::read)
        .map_err(io_failure("runner-identity"))?;
    let current_executable_sha256 = sha256_hex(&current_executable);
    if current_executable_sha256 != manifest.runner.sha256 {
        return Err((
            "runner-identity".to_owned(),
            "runner_identity_mismatch".to_owned(),
            format!(
                "当前执行文件 SHA-256 {} 与 bundle runner {} 不同",
                current_executable_sha256, manifest.runner.sha256
            ),
        ));
    }

    let model_manifest_path = root.join(&manifest.model_manifest.path);
    let model_json = std::fs::read_to_string(&model_manifest_path).map_err(|error| {
        (
            "manifest-parse".to_owned(),
            "model_manifest_unreadable".to_owned(),
            error.to_string(),
        )
    })?;
    let model_manifest = ModelManifest::parse_and_validate(&model_json).map_err(|error| {
        (
            "manifest-parse".to_owned(),
            error.code().to_owned(),
            error.to_string(),
        )
    })?;
    let artifact = model_manifest
        .artifacts
        .iter()
        .find(|candidate| candidate.sha256 == manifest.artifact.sha256)
        .ok_or_else(|| {
            (
                "artifact-integrity".to_owned(),
                "artifact_identity_mismatch".to_owned(),
                "bundle artifact 未被 model manifest 以相同 SHA-256 引用".to_owned(),
            )
        })?;
    let factory =
        AndroidLiteRtV2Factory::new(model_manifest.clone(), root, AndroidLiteRtAccelerator::Cpu);
    let request = BackendInitRequest {
        target: manifest.target.clone(),
        model_id: model_manifest.model.id.clone(),
        model_version: model_manifest.model.version.clone(),
        artifact_id: artifact.id.clone(),
        artifact_sha256: artifact.sha256.clone(),
    };
    let initialized = Instant::now();
    let instance = factory.create(&request).map_err(|error| {
        (
            format!("{:?}", error.stage),
            error.code.to_string(),
            error.message.to_string(),
        )
    })?;
    let initialization_ms = initialized
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    report.performance.initialization_ms = initialization_ms;
    if initialization_ms > manifest.gates.initialization_deadline_ms {
        return Err((
            "initialization-timeout".to_owned(),
            "native_initialization_timeout".to_owned(),
            format!(
                "初始化耗时 {initialization_ms} ms，超过 {} ms",
                manifest.gates.initialization_deadline_ms
            ),
        ));
    }
    report.resolved = Some(instance.resolved.clone());
    report.io_diagnostics = serde_json::to_value(instance.backend.diagnostics()).ok();
    let mut backend = instance.backend;
    let output_root = root.join("outputs");
    std::fs::create_dir_all(&output_root).map_err(io_failure("output-create"))?;

    for fixture in &manifest.fixtures {
        let input_bytes =
            std::fs::read(root.join(&fixture.input.path)).map_err(io_failure("input-read"))?;
        for run_index in 1..=manifest.gates.golden_runs {
            let started = Instant::now();
            let output = backend
                .infer(ModelInput::Tensor {
                    role: fixture.role.clone(),
                    shape: fixture.shape.clone(),
                    dtype: fixture.dtype,
                    bytes: input_bytes.clone(),
                })
                .map_err(|error| ("inference".to_owned(), error.code, error.message))?;
            let inference_ms = started.elapsed().as_secs_f64() * 1000.0;
            if report.performance.cold_inference_ms.is_none() {
                report.performance.cold_inference_ms = Some(inference_ms);
            }
            let tensor = output.tensors.into_iter().next().ok_or_else(|| {
                (
                    "output".to_owned(),
                    "missing_output".to_owned(),
                    "LiteRT 未返回输出".to_owned(),
                )
            })?;
            let dtype = tensor_dtype(&tensor.data);
            let bytes = tensor_bytes(&tensor.data);
            let relative = format!("outputs/{}/run-{run_index}.bin", fixture.id);
            let path = root.join(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).map_err(io_failure("output-create"))?;
            std::fs::write(&path, &bytes).map_err(io_failure("output-write"))?;
            report.outputs.push(AndroidRunnerOutput {
                fixture_id: fixture.id.clone(),
                fixture_kind: fixture.kind,
                run: run_index,
                path: relative,
                bytes: bytes.len() as u64,
                sha256: sha256_hex(&bytes),
                shape: tensor.shape,
                dtype,
                inference_ms,
            });
        }
    }

    if let Some(fixture) = manifest.fixtures.first() {
        let input_bytes = std::fs::read(root.join(&fixture.input.path))
            .map_err(io_failure("performance-input-read"))?;
        for _ in 0..manifest.gates.performance_warmup_runs {
            backend
                .infer(ModelInput::Tensor {
                    role: fixture.role.clone(),
                    shape: fixture.shape.clone(),
                    dtype: fixture.dtype,
                    bytes: input_bytes.clone(),
                })
                .map_err(|error| ("warmup".to_owned(), error.code, error.message))?;
        }
        for _ in 0..manifest.gates.performance_sample_runs {
            let started = Instant::now();
            backend
                .infer(ModelInput::Tensor {
                    role: fixture.role.clone(),
                    shape: fixture.shape.clone(),
                    dtype: fixture.dtype,
                    bytes: input_bytes.clone(),
                })
                .map_err(|error| ("performance".to_owned(), error.code, error.message))?;
            report
                .performance
                .warm_inference_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    report.performance.peak_process_rss_bytes = peak_rss_bytes();
    report.loaded_libraries = loaded_libraries();
    let expected_runtime_sha256 = &manifest.runtime_libraries[0].sha256;
    if !report.loaded_libraries.iter().any(|library| {
        library.path.ends_with("libLiteRt.so")
            && library.sha256.as_ref() == Some(expected_runtime_sha256)
    }) {
        return Err((
            "package-load".to_owned(),
            "litert_library_identity_not_mapped".to_owned(),
            "/proc/self/maps 中未找到 digest 与 bundle 一致的 libLiteRt.so".to_owned(),
        ));
    }
    Ok(())
}

fn base_report(manifest: &AndroidBundleManifest, manifest_sha256: String) -> AndroidRunnerReport {
    AndroidRunnerReport {
        schema_version: ANDROID_RUNNER_REPORT_SCHEMA_VERSION,
        state: AndroidRunnerReportState::Failed,
        selection_code: "runner_not_started".to_owned(),
        failure_stage: None,
        failure_message: None,
        bundle_id: manifest.bundle_id.clone(),
        bundle_manifest_sha256: manifest_sha256,
        target: manifest.target.clone(),
        runner_sha256: manifest.runner.sha256.clone(),
        model_manifest_sha256: manifest.model_manifest.sha256.clone(),
        artifact_sha256: manifest.artifact.sha256.clone(),
        runtime_version: LITERT_RUNTIME_VERSION.to_owned(),
        rust_binding_version: LITERT_RUST_BINDING_VERSION.to_owned(),
        configured_provider: "LiteRT CompiledModel".to_owned(),
        accelerator: "CPU".to_owned(),
        resolved: None,
        io_diagnostics: None,
        device: AndroidRunnerDevice {
            serial: getprop("ro.serialno"),
            manufacturer: getprop("ro.product.manufacturer"),
            model: getprop("ro.product.model"),
            fingerprint: getprop("ro.build.fingerprint"),
            api_level: getprop("ro.build.version.sdk"),
            abi: getprop("ro.product.cpu.abi"),
        },
        outputs: Vec::new(),
        performance: AndroidRunnerPerformance {
            initialization_ms: 0,
            initialization_deadline_ms: manifest.gates.initialization_deadline_ms,
            cold_inference_ms: None,
            warmup_runs: manifest.gates.performance_warmup_runs,
            warm_inference_ms: Vec::new(),
            peak_process_rss_bytes: None,
        },
        loaded_libraries: Vec::new(),
    }
}

fn tensor_bytes(data: &TensorData) -> Vec<u8> {
    match data {
        TensorData::F32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TensorData::F16(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TensorData::I8(values) => values.iter().map(|value| *value as u8).collect(),
        TensorData::U8(values) => values.clone(),
        TensorData::I32(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TensorData::I64(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        TensorData::Bool(values) => values.iter().map(|value| u8::from(*value)).collect(),
    }
}

fn tensor_dtype(data: &TensorData) -> rimeflow_onnx_base::DType {
    match data {
        TensorData::F32(_) => rimeflow_onnx_base::DType::F32,
        TensorData::F16(_) => rimeflow_onnx_base::DType::F16,
        TensorData::I8(_) => rimeflow_onnx_base::DType::I8,
        TensorData::U8(_) => rimeflow_onnx_base::DType::U8,
        TensorData::I32(_) => rimeflow_onnx_base::DType::I32,
        TensorData::I64(_) => rimeflow_onnx_base::DType::I64,
        TensorData::Bool(_) => rimeflow_onnx_base::DType::Bool,
    }
}

fn getprop(name: &str) -> Option<String> {
    std::process::Command::new("/system/bin/getprop")
        .arg(name)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

fn loaded_libraries() -> Vec<AndroidLoadedLibrary> {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return Vec::new();
    };
    maps.lines()
        .filter_map(|line| line.split_whitespace().last())
        .filter(|path| path.starts_with('/') && path.ends_with(".so"))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|path| AndroidLoadedLibrary {
            sha256: std::fs::read(&path).ok().map(|bytes| sha256_hex(&bytes)),
            path,
        })
        .collect()
}

fn io_failure(stage: &'static str) -> impl FnOnce(std::io::Error) -> (String, String, String) {
    move |error| {
        (
            stage.to_owned(),
            "runner_io_failed".to_owned(),
            error.to_string(),
        )
    }
}

fn required_option(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("缺少参数 {name}"))
}
