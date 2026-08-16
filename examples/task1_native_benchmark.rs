use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use rimeflow_onnx_base::native_ort::NativeOrtBackend;
use serde_json::json;
use sha2::{Digest, Sha256};

const WARMUP_RUNS: usize = 5;
const SAMPLE_RUNS: usize = 30;

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn percentile(samples: &[f64], percentile: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * percentile).ceil() as usize;
    sorted[index]
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|value| value * 1024)
}

#[cfg(target_os = "windows")]
fn peak_rss_bytes() -> Option<u64> {
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>()
            .try_into()
            .ok()?,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let succeeded =
        unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (succeeded != 0).then_some(counters.peak_working_set_size as u64)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

fn digest_f32(values: &[f32]) -> String {
    format!("{:x}", Sha256::digest(bytemuck::cast_slice(values)))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let model_path = PathBuf::from(args.next().ok_or("model path missing")?);
    let input_path = PathBuf::from(args.next().ok_or("input path missing")?);
    let report_path = PathBuf::from(args.next().ok_or("report path missing")?);
    let output_path = PathBuf::from(args.next().ok_or("output path missing")?);
    let model = fs::read(&model_path)?;
    let input_bytes = fs::read(&input_path)?;
    let input: &[f32] = bytemuck::try_cast_slice(&input_bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    if input.len() != 3 * 640 * 640 || input.iter().any(|value| !value.is_finite()) {
        return Err("canonical input must contain 1x3x640x640 finite float32 values".into());
    }

    let wgpu_start = Instant::now();
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let adapter_info = adapter.get_info();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let wgpu_initialization_ms = milliseconds(wgpu_start.elapsed());

    let initialization_start = Instant::now();
    let mut backend = NativeOrtBackend::new(&model, &device, &queue, 640)?;
    let initialization_ms = milliseconds(initialization_start.elapsed());

    let cold_start = Instant::now();
    let mut output = backend.infer_from_host_slice(input)?;
    let cold_inference_ms = milliseconds(cold_start.elapsed());
    for _ in 0..WARMUP_RUNS {
        output = backend.infer_from_host_slice(input)?;
    }
    let mut warm_samples = Vec::with_capacity(SAMPLE_RUNS);
    for _ in 0..SAMPLE_RUNS {
        let start = Instant::now();
        output = backend.infer_from_host_slice(input)?;
        warm_samples.push(milliseconds(start.elapsed()));
    }
    if output.len() != 84 * 8400 || output.iter().any(|value| !value.is_finite()) {
        return Err("native output must contain 1x84x8400 finite float32 values".into());
    }
    fs::write(&output_path, bytemuck::cast_slice(&output))?;
    let report = json!({
        "schemaVersion": 1,
        "runtime": { "name": "rimeflow-onnx-base::NativeOrtBackend", "ortCrate": "2.0.0-rc.12", "resolvedExecutionProvider": format!("{:?}", backend.resolved_ep()) },
        "adapter": { "classification": "wgpu-preprocess-adapter-not-ort-execution-provider", "name": adapter_info.name, "backend": format!("{:?}", adapter_info.backend), "driver": adapter_info.driver, "driverInfo": adapter_info.driver_info },
        "method": { "input": "infer_from_host_slice", "warmupRuns": WARMUP_RUNS, "sampleRuns": SAMPLE_RUNS },
        "metrics": { "wgpuInitializationMs": wgpu_initialization_ms, "initializationMs": initialization_ms, "initializationExcludes": "wgpuInitializationMs", "coldInferenceMs": cold_inference_ms, "warmInferenceMs": { "p50": percentile(&warm_samples, 0.50), "p95": percentile(&warm_samples, 0.95), "samples": warm_samples }, "peakProcessRssBytes": peak_rss_bytes() },
        "artifacts": { "modelBytes": model.len(), "canonicalInputBytes": input_bytes.len(), "outputElements": output.len(), "outputFiniteCount": output.len(), "outputSha256Float32Le": digest_f32(&output) },
    });
    fs::write(
        report_path,
        format!("{}\n", serde_json::to_string_pretty(&report)?),
    )?;
    Ok(())
}
