# rimeflow-onnx-base

Cross-target infrastructure for the **rimeflow** family of ONNX operator
crates. Operator crates (`rimeflow-yolov8n`, `rimeflow-siglip`,
`rimeflow-sam2`, `rimeflow-depth-anything`, …) depend on this and only own
the model-specific bits:

- `shaders/preprocess.wgsl` — the compute shader for letterbox + normalize
- `models/*.onnx` — model weights
- `src/postprocess.rs` — model-specific decode + NMS + result type
- A newtype / re-export layer over this crate's types

Everything else — the wgpu compute pipeline, letterbox math, `pykeio/ort` v2
session with EP fan-out (CoreML/DirectML/CUDA/TensorRT/NNAPI/QNN/OpenVINO,
CPU fallback), the `onnxruntime-web` inline_js bridge, wgpu HAL escape hatch
helpers, and the `build.rs` codegen — lives here.

## Public surface

| Module                                    | Purpose                                                                                              |
|-------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`preprocess`]                            | `PreprocessPipeline`, `LetterboxParams`, `PreprocessUniform`                                         |
| [`native_ort`] (cfg + `feature=native`)   | `NativeOrtBackend`, `ResolvedEp`, `InferError`                                                       |
| [`ort_bridge`] (cfg=wasm32)               | `extract_web_gpu_buffer`, `get_output_f32`, `get_f64` — helpers around the operator-generated extern |
| [`build_helper`] (`feature=build-helper`) | `generate_extern_block(&BridgeConfig)` — called from operator crates' own `build.rs`                 |

## Adding a new operator crate

1. `cargo new --lib rimeflow-<model>`.
2. Depend on this crate twice — once as a normal dep, once as a build-dep with
   `features = ["build-helper"]`:

   ```toml
   [dependencies]
   rimeflow-onnx-base = { git = "https://github.com/caozisheng/rimeflow-onnx-base", branch = "main" }
   wgpu     = { version = "29", default-features = false, features = ["wgsl"] }
   bytemuck = { version = "1", features = ["derive"] }

   [build-dependencies]
   rimeflow-onnx-base = { git = "https://github.com/caozisheng/rimeflow-onnx-base", branch = "main", features = ["build-helper"] }

   [features]
   native            = ["rimeflow-onnx-base/native"]
   native-directml   = ["native", "rimeflow-onnx-base/native-directml"]
   # ... one line per EP ...
   model-embedded    = []
   ```

3. Drop `shaders/preprocess.wgsl` in place. It MUST bind the standard slots
   (see `preprocess::PreprocessPipeline` docs).
4. Write `src/postprocess.rs` with a `Detection` (or model-appropriate
   result) type and decode + NMS logic.
5. `build.rs`:

   ```rust
   use rimeflow_onnx_base::build_helper::{generate_extern_block, BridgeConfig};
   use std::path::Path;

   fn main() {
       generate_extern_block(&BridgeConfig {
           wgsl_path: Path::new("shaders/preprocess.wgsl"),
           dst_size:  640,
       }).unwrap();
   }
   ```

6. `src/lib.rs`:

   ```rust
   pub use rimeflow_onnx_base::{LetterboxParams, PreprocessPipeline};

   pub mod postprocess;
   #[cfg(all(not(target_arch = "wasm32"), feature = "native"))]
   pub mod native_ort { pub use rimeflow_onnx_base::native_ort::*; }
   #[cfg(target_arch = "wasm32")]
   pub mod ort_bridge {
       pub use rimeflow_onnx_base::ort_bridge::*;
       include!(concat!(env!("OUT_DIR"), "/ort_bridge_generated.rs"));
   }

   pub const MODEL_URL: &str = "/models/<model>.onnx";
   #[cfg(feature = "model-embedded")]
   pub const MODEL_BYTES: &[u8] = include_bytes!("../models/<model>.onnx");
   pub const DST_SIZE: u32 = 640;
   pub const INPUT_SHAPE: [i64; 4] = [1, 3, 640, 640];
   ```

## Design docs

See the RimeCut monorepo's `docs/rimecut-feature-dev-rules.md` — §5 covers
the operator crate template and §16 is the full base-crate split design.

## License

MIT. See [LICENSE](LICENSE).
