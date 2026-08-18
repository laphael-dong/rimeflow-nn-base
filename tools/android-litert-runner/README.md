# Android LiteRT v2 原生 runner

该目录提供可复现的 Android API 26+、CPU-only LiteRT 2.1.6 原生 runner，单次构建严格选择 `arm64-v8a` 或 `x86_64` 一个 ABI。它直接调用 Base 的 `AndroidLiteRtV2Factory` 与 production `CompiledModel` 路径；交叉编译成功只表示 `build-verified`，不表示真机 conformance、Android supported 状态或 RimeCut 产品验收。

## 冻结依赖

- Android NDK r27c：`27.2.12479018`，Linux zip SHA-256 `59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc`，仅下载到仓库忽略的 `.android-litert/`。
- LiteRT C++ SDK 2.1.6：SHA-256 `2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8`。
- Android arm64 `libLiteRt.so` 2.1.6：SHA-256 `35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369`。
- Android x86_64 `libLiteRt.so` 2.1.6：SHA-256 `aa1530ba8b37b537d37139760716d183d2d7dc1f7781791ddf1d071c73eca535`。
- Rust binding `google-ai-edge-litert` 0.1.3：官方 crate SHA-256 `fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8`；仓库内 vendor 副本修补可复现 build path，并让 `CompiledModel` 按 signature index 取得真实 tensor。修补后禁止 build script 联网，并强制 CMake、bindgen、Rust linker 使用 NDK r27c API 26 工具。
- Rust/Cargo 固定为 `1.97.1`，依赖解析固定在仓库 `Cargo.lock`，registry/cache 写入忽略的 `.android-litert/cargo-home/`；CMake 验证版本为 `3.28.3`。

## 构建 bundle

模型、conversion manifest 与 fixture tensor 均保持外部、digest-addressed，不提交 AGPL fixture 或 TFLite 字节。先按 `fixtures.example.json` 创建一个位于忽略目录的 fixture 配置，再运行：

```bash
tools/android-litert-runner/build_bundle.sh \
  --arch x86_64 \
  --artifact /absolute/path/to/yolov8n-fp32.tflite \
  --conversion-manifest /absolute/path/to/litert-artifact-manifest.json \
  --conversion-manifest-sha256 f1f95bac2006cd02e364123ab9e7556cc331e6d20f49006c50fcebd15d0c4881 \
  --fixtures /absolute/path/to/fixtures.json \
  --out "$PWD/.android-litert/bundles"
```

`--arch` 仅接受 `arm64`（默认，保持既有 producer 行为）或 `x86_64`。下游可在不下载、不编译的情况下读取同一锁定契约：

```bash
tools/android-litert-runner/build_bundle.sh --arch x86_64 --print-contract
```

输出包含 Cargo target、NDK compiler target、CMake ABI、bundle/provenance target，以及 LiteRT runtime URL 和 SHA-256。packager 会再次逐字段验证该 provenance，并检查 runner 与 runtime 的 ELF machine；ARM64 bytes、目标不一致或未批准架构都会被拒绝。

构建器先校验下载 digest，再构建 runner，最后生成 `sha256-<content-id>/`。可单独执行自检：

```bash
node tools/android-litert-runner/package_bundle.mjs --verify .android-litert/bundles/sha256-<content-id>
```

## LiteRT signature I/O 映射

LiteRT 2.1.6 区分 signature binding name 和 subgraph tensor name。比如 signature 的输入 binding 可以是 `args_0`，但同一 signature position 0 指向的实际 tensor 名称可以是 `serving_default_args_0`；binding name 不能传给 `Subgraph::input_tensor_by_name`。

runner 按固定规则解析两种身份：先枚举 signature binding name 并验证 arity/唯一性，再通过 `LiteRtGetSignatureInputTensorByIndex` 或 `LiteRtGetSignatureOutputTensorByIndex` 获取该 position 的真实 tensor；最后用实际 tensor name、signature position、dtype 和 runtime shape 对 manifest 做唯一且完整的 AND 匹配。机器诊断中的 `runtimeName` 是实际 tensor name，`signatureBindingName` 单独保存 signature binding；缺失、重复、歧义或 dtype/shape 不一致均在 `IoDiscovery` 阶段失败。

## bundle 布局

```text
sha256-<content-id>/
  bundle-manifest.json
  bundle-manifest.sha256
  bin/rimeflow-android-litert-runner
  lib/libLiteRt.so
  provenance/android-runner-build.json
  provenance/litert-artifact-manifest.json
  manifest/model-manifest.json
  model/yolov8n-fp32.tflite
  inputs/<fixture>.f32le.bin
  outputs/
  reports/
```

设备管理员以后可把整个 digest 目录作为一个资源单元部署，并以同目录动态库搜索路径运行：

```bash
LD_LIBRARY_PATH="$BUNDLE/lib" "$BUNDLE/bin/rimeflow-android-litert-runner" \
  --bundle "$BUNDLE" --report "$BUNDLE/reports/runner-report.json"
```

runner 在 runtime 初始化前校验 manifest 列出的 runner、SDK provenance、runtime、model manifest、TFLite 与所有输入。报告包含稳定 selection/failure code、设备属性、adapter/I/O identity、初始化/cold/warm timing、peak RSS hook、原始输出 digest，以及从 `/proc/self/maps` 采集的实际 shared-library load identity；`fault-never-promote` fixture 永远保持独立标签，不可作为 golden promotion 输入。

该报告可供后续设备任务映射到 `AdapterConformanceReport` 的八项 check，但本任务不修改历史 blocked report，也不把 build-only 结果转换为 `real-target` 或 `passed`。
