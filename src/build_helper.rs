//! Build-script helper used by operator crates' `build.rs`.
//!
//! wasm-bindgen's `#[wasm_bindgen(inline_js = "...")]` needs a **string
//! literal at macro expansion time** — that means the extern block can't be
//! generic over an operator's `MODEL_URL` / `DST_SIZE` / `PREPROCESS_WGSL`.
//! Every operator crate must therefore generate its own extern block. This
//! module owns the JS template and the substitution + Rust codegen so
//! operators only see a 5-line `build.rs`.
//!
//! # Example
//!
//! ```ignore
//! // operator crate `rimeflow-yolov8n/build.rs`
//! use rimeflow_onnx_base::build_helper::{generate_extern_block, BridgeConfig};
//! use std::path::Path;
//!
//! fn main() {
//!     generate_extern_block(&BridgeConfig {
//!         wgsl_path:  Path::new("shaders/preprocess.wgsl"),
//!         dst_size:   640,
//!     }).unwrap();
//! }
//! ```
//!
//! # Contract with the operator crate
//!
//! - Operator crate declares `[build-dependencies] rimeflow-onnx-base = { …,
//!   features = ["build-helper"] }`.
//! - Operator crate's `src/ort_bridge.rs` contains only
//!   `include!(concat!(env!("OUT_DIR"), "/ort_bridge_generated.rs"));`.
//! - The generated file emits a full `#[wasm_bindgen(inline_js = ...)]`
//!   extern block declaring `capture_webgpu_device`, `ort_init`,
//!   `ort_run_gpu_buffer`, `ort_run_cpu_slice`, `ort_detect`, `ort_release`.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::backend::Platform;
use crate::manifest::{Artifact, ManifestError, ModelManifest};

/// Configuration for `generate_extern_block`.
pub struct BridgeConfig<'a> {
    /// Path to the operator's WGSL shader, relative to the operator crate's
    /// Cargo manifest directory (e.g. `Path::new("shaders/preprocess.wgsl")`).
    pub wgsl_path: &'a Path,
    /// Operator input tile size (square). Substituted into the JS
    /// `const DST_SIZE = …;`.
    pub dst_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedBuildArtifact {
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub artifact: Artifact,
}

#[derive(Debug)]
pub enum BuildManifestError {
    Io(std::io::Error),
    Manifest(ManifestError),
}

impl fmt::Display for BuildManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "manifest build I/O: {error}"),
            Self::Manifest(error) => write!(formatter, "manifest build validation: {error}"),
        }
    }
}

impl std::error::Error for BuildManifestError {}

impl From<std::io::Error> for BuildManifestError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ManifestError> for BuildManifestError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Parse a manifest, select a target artifact, and verify the artifact bytes.
///
/// Operator build scripts can call this before embedding model data. The
/// artifact path is resolved relative to the manifest file, and both inputs
/// are registered with Cargo's incremental rebuild tracking.
pub fn validate_manifest_artifact(
    manifest_path: &Path,
    artifact_id: &str,
    target: &Platform,
) -> Result<ValidatedBuildArtifact, BuildManifestError> {
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let manifest_json = fs::read_to_string(manifest_path)?;
    let manifest = ModelManifest::parse_and_validate(&manifest_json)?;
    let artifact = manifest.select_artifact(artifact_id, target)?.clone();
    let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let artifact_path = parent.join(&artifact.path);
    println!("cargo:rerun-if-changed={}", artifact_path.display());
    let bytes = fs::read(&artifact_path)?;
    ModelManifest::verify_artifact_bytes(&artifact, &bytes)?;
    Ok(ValidatedBuildArtifact {
        manifest_path: manifest_path.to_path_buf(),
        artifact_path,
        artifact,
    })
}

/// Read the base's `template.js` and the operator's WGSL, substitute
/// sentinels, and emit `${OUT_DIR}/ort_bridge_generated.rs` — a self-contained
/// `#[wasm_bindgen(inline_js = "...")]` extern block.
///
/// Also emits the appropriate `cargo:rerun-if-changed=` lines so builds are
/// incremental.
///
/// For non-wasm32 targets this writes a placeholder file so that stray
/// `include!(concat!(env!("OUT_DIR"), "/ort_bridge_generated.rs"))` in cfg-gated
/// modules compiles cleanly.
pub fn generate_extern_block(cfg: &BridgeConfig<'_>) -> std::io::Result<()> {
    println!("cargo:rerun-if-changed={}", cfg.wgsl_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR unset"));
    let out_rs = out_dir.join("ort_bridge_generated.rs");

    // Non-wasm build: emit an empty stub so operator crate's non-wasm build
    // succeeds even if it accidentally references the file (it shouldn't —
    // the `include!` sits behind `cfg(target_arch = "wasm32")`).
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "wasm32" {
        fs::write(
            &out_rs,
            "// non-wasm target; ort_bridge module is cfg-out.\n",
        )?;
        return Ok(());
    }

    let wgsl = fs::read_to_string(cfg.wgsl_path).unwrap_or_else(|e| {
        panic!(
            "rimeflow-onnx-base build_helper: failed to read {}: {}",
            cfg.wgsl_path.display(),
            e,
        )
    });
    // The template lives inside this crate; embed it at compile time so
    // operator crates don't have to know where the base's checkout is.
    let template = TEMPLATE_JS;

    // Escape WGSL for a JS template literal (`…`).
    let wgsl_js = wgsl
        .replace('\\', "\\\\")
        .replace('`', "\\`")
        .replace("${", "\\${");

    // Sentinels: single occurrence each (asserted below). Template comments
    // never repeat the raw sentinel string.
    let sent_wgsl = "__PREPROCESS_WGSL__";
    let sent_dst = "__DST_SIZE__";
    let n_wgsl = template.matches(sent_wgsl).count();
    let n_dst = template.matches(sent_dst).count();
    assert_eq!(
        n_wgsl, 1,
        "rimeflow-onnx-base template.js must contain exactly one `{sent_wgsl}`; found {n_wgsl}",
    );
    assert_eq!(
        n_dst, 1,
        "rimeflow-onnx-base template.js must contain exactly one `{sent_dst}`; found {n_dst}",
    );

    let js = template
        .replace(sent_wgsl, &wgsl_js)
        .replace(sent_dst, &cfg.dst_size.to_string());

    // Encode the entire JS blob as a Rust raw string literal for
    // `#[wasm_bindgen(inline_js = r#"..."#)]`. Grow the `#`-count until no
    // interior `"###...` sequence collides with the delimiter.
    let mut hashes = 1usize;
    loop {
        let needle = format!("\"{}", "#".repeat(hashes));
        if !js.contains(&needle) {
            break;
        }
        hashes += 1;
    }
    let d = "#".repeat(hashes);

    // Assemble the Rust source by string concatenation (avoiding format!'s
    // `{}` clashing with JS syntax).
    let mut out = String::new();
    out.push_str("// Auto-generated by rimeflow_onnx_base::build_helper — do not edit.\n");
    out.push_str("// The inline_js payload is template.js with the operator's\n");
    out.push_str("// preprocess.wgsl substituted for __PREPROCESS_WGSL__ and the\n");
    out.push_str("// operator's DST_SIZE substituted for __DST_SIZE__.\n\n");
    out.push_str("use wasm_bindgen::prelude::*;\n\n");

    out.push_str("#[wasm_bindgen(inline_js = r");
    out.push_str(&d);
    out.push('"');
    out.push_str(&js);
    out.push('"');
    out.push_str(&d);
    out.push_str(")]\n");

    out.push_str("extern \"C\" {\n");
    out.push_str("    pub fn capture_webgpu_device();\n\n");

    out.push_str("    #[wasm_bindgen(catch)]\n");
    out.push_str("    pub async fn ort_init(model_url: &str) -> Result<JsValue, JsValue>;\n\n");

    out.push_str("    /// Main-line API: hand in a `GPUBuffer` extracted from a\n");
    out.push_str("    /// `wgpu::Buffer` (same GPUDevice). Returns the raw model output\n");
    out.push_str("    /// tensor as a JS `Float32Array` (wrapped in `JsValue`).\n");
    out.push_str("    #[wasm_bindgen(catch)]\n");
    out.push_str("    pub async fn ort_run_gpu_buffer(\n");
    out.push_str("        gpu_buffer: &web_sys::GpuBuffer,\n");
    out.push_str("        dims: js_sys::Array,\n");
    out.push_str("    ) -> Result<JsValue, JsValue>;\n\n");

    out.push_str("    /// Fallback API: hand in a host `Float32Array`.\n");
    out.push_str("    #[wasm_bindgen(catch)]\n");
    out.push_str("    pub async fn ort_run_cpu_slice(\n");
    out.push_str("        nchw: &js_sys::Float32Array,\n");
    out.push_str("        dims: js_sys::Array,\n");
    out.push_str("    ) -> Result<JsValue, JsValue>;\n\n");

    out.push_str("    /// Transition API for open_quartz and other canvas-based debug\n");
    out.push_str("    /// tools. Main-line callers MUST NOT use this. Returns a JS dict\n");
    out.push_str("    /// `{ output: Float32Array, scale, padX, padY, srcW, srcH }`.\n");
    out.push_str("    #[wasm_bindgen(catch)]\n");
    out.push_str("    pub async fn ort_detect(\n");
    out.push_str("        canvas: &web_sys::HtmlCanvasElement,\n");
    out.push_str("    ) -> Result<JsValue, JsValue>;\n\n");

    out.push_str("    pub fn ort_release();\n");
    out.push_str("}\n");

    fs::write(&out_rs, out)?;
    Ok(())
}

/// The JS template embedded at compile time. Exposed for testing / inspection.
pub const TEMPLATE_JS: &str = include_str!("../template.js");
