# RFB-ANDROID-RUNNER-01 实现报告

## 结论与边界

本任务在 Base 中实现了 Android `arm64-v8a`、API 26+、CPU-only LiteRT v2 原生 runner、可复现构建器、digest-scoped bundle 和机器报告合约。最终状态仅为 `build-verified`：未向 RFCX90C568W 或任何手机写入内容，未运行真机推理，未修改历史 blocked conformance report、Android supported 状态、RimeCut、阈值或 OpenSpec ledger，也未声明 GPU/NPU、产品或正式设备验收。

- 基线：`aaff04fa416b0366d6e69a61b4887b6d2fdd4215`
- 分支：`rfb-android-runner-01`
- 交叉目标：`aarch64-linux-android` / `arm64-v8a` / API 26 / CPU
- production path：`AndroidLiteRtV2Factory` -> `Android CompiledModel` -> `RuntimeBackend::infer`
- 外部模型和 fixture 只复制到忽略的 `.android-litert/`，没有提交 AGPL fixture 或 TFLite 字节。

## 冻结输入与工具

| 输入 | 版本/身份 | URL / SHA-256 |
| --- | --- | --- |
| Rust | `rustc 1.97.1 (8bab26f4f 2026-07-14)` | 构建器精确检查 |
| Cargo | `cargo 1.97.1 (c980f4866 2026-06-30)` | `Cargo.lock` + 任务本地 `.android-litert/cargo-home` |
| CMake | `3.28.3` | 记录到 build provenance |
| Android NDK | r27c / `27.2.12479018` | `https://dl.google.com/android/repository/android-ndk-r27c-linux.zip` / `59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc` |
| LiteRT C++ SDK | `2.1.6` | `https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_cc_sdk.zip` / `2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8` |
| LiteRT runtime | Android arm64 `2.1.6` | `https://storage.googleapis.com/litert/binaries/2.1.6/android_arm64/libLiteRt.so` / `35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369` |
| Rust binding | `google-ai-edge-litert 0.1.3` | 官方 crate / `fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8` |
| TFLite | 12,841,227 bytes | `794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1` |
| conversion manifest | schema v1 | `f1f95bac2006cd02e364123ab9e7556cc331e6d20f49006c50fcebd15d0c4881` |

所有下载先写任务本地 `.partial`，通过固定 SHA-256 后才进入缓存。bundle 内 `provenance/android-runner-build.json` 机器可读地保留 URL、版本和 digest；runner 初始化前再次验证 manifest 所列 runner、runtime、provenance、model manifest、TFLite 与每个输入。

## 上游 binding 审计

官方 `0.1.3` build script 会在 Cargo build 期间直接访问可变下载位置、禁用总下载超时、使用宿主 `clang/clang++`、构建整个 SDK 示例目标，并输出全部环境变量。vendored 最小修补执行以下约束：

- build script 不再联网，只接收调用方预下载的 SDK/runtime，并在复制前重新计算 SHA-256。
- CMake toolchain、C/C++ compiler、bindgen target/sysroot 和 Rust linker 均指向任务本地 NDK r27c `aarch64-linux-android26`。
- 仅构建 binding 实际需要的 `litert_cc_api`，不构建与 runner 无关且需额外 Android `liblog` 的 `dump_model_simple`。
- 禁用环境转储，避免构建日志泄露无关凭据；CPU build 保持 `LITERT_DISABLE_GPU`。
- `[patch.crates-io]` 仅替换该固定版本的本地 build path，没有替换运行时 API 或扩大依赖闭包。

## 构建与修复周期

1. 首次交叉构建成功配置 NDK r27c CMake 并构建 `liblitert_cc_api.a`，但 bindgen 找不到 libclang。诊断确认 r27c 的库位于 `toolchains/llvm/prebuilt/linux-x86_64/musl/lib`，修正 `LIBCLANG_PATH` 后复测。
2. 复测越过 bindgen，失败于 SDK 自带 `dump_model_simple` 未链接 `__android_log_vprint`。将 CMake target 收窄为生产依赖 `litert_cc_api` 后，Android release compile/link/package 成功；有界修复预算至此用完且不再出现目标构建错误。

后续 provenance shell 续行转义错误发生在编译前、没有外部副作用，修正后通过 `bash -n` 和最终构建。完整无环境转储的首次任务本地 Cargo 构建日志保留在 `.android-litert/build-and-package-final-2.log`，当前源的最终目标复核日志保留在 `.android-litert/android-final-check.log`，均为 worktree-local ignored evidence。

## 最终 bundle 证据

- bundle：`sha256-dfd7ffebf87f7c120bb0802cb764f2de0d3f7c1c0bbbaf5119081e64a6e6b807`
- bundle manifest SHA-256：`5c822c266156c326ed9adee3c32f7cc93c7bcca00b40145622699d89c0918aef`
- runner SHA-256：`6af2fd5a062116f6c8a61bb6a9ea5f395b242ec11dceab599051e88cb59fe7a1`
- runtime SHA-256：`35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369`
- 五个历史 fixture：`no-detection`、`single-target`、`multi-class`、`boundary-box`、`extreme-aspect`，均为 NCHW FP32 `[1,3,640,640]`、4,915,200 bytes，并逐项验证既有 SHA-256。
- ELF：64-bit little-endian PIE、Machine `AArch64`、interpreter `/system/bin/linker64`。
- dynamic dependencies：`libdl.so`、`libLiteRt.so`、`libc.so`；无 RPATH/RUNPATH，部署命令显式设置 bundle-local `LD_LIBRARY_PATH`。

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
  outputs/<fixture>/run-<n>.bin
  reports/runner-report.json
```

`package_bundle.mjs --verify` 校验每个文件、manifest sidecar、model-to-artifact identity、冻结 build provenance，以及 bundle ID/目录名对内容身份的绑定。runner 还校验当前执行文件 digest，并要求 `/proc/self/maps` 中实际映射的 `libLiteRt.so` digest 与 bundle runtime 完全一致。

## 运行时证据接口

runner 报告固定记录 runtime/binding/adapter/artifact/runner identity、设备 `getprop` 信息、I/O diagnostics、初始化耗时、首个 cold inference、5 次 warmup、30 次 warm sample、peak RSS、loaded library 路径与 digest、每个 fixture 两次 raw output 的 shape/dtype/bytes/SHA-256。fixture kind 区分 `golden` 与 `fault-never-promote`，后者不会被后续工具误作 golden promotion 输入；该结构可映射到 `AdapterConformanceReport` 的 manifest I/O、timeout、smoke、golden、fault、diagnostics、performance 和 package-load 八项检查，但本任务不生成伪造的 real-target report。

## 验证结果

- `cargo test android_runner --lib`：3 passed。
- `cargo test --test litert_v2_adapter`：5 passed。
- `cargo test --test platform_factory`：5 passed。
- `cargo test`：全部默认 host tests passed；1 个 doc test ignored，0 failed。
- `cargo fmt --all -- --check`：passed（最终格式化后）。
- `bash -n tools/android-litert-runner/build_bundle.sh`：passed。
- `node --check tools/android-litert-runner/package_bundle.mjs`：passed。
- 两份新 JSON schema 可解析；最终 bundle semantic/self-check：passed。
- NDK r27c Android release compile/link：passed；最终 `file`/`readelf` 证据符合 AArch64 Android CPU bundle。
- `git diff --check`：passed。

## 残余风险

- 没有设备执行证据；初始化、推理输出、RSS、package-load 与设备身份只能由后续获授权的 RFCX90C568W 任务确认，当前不得提升 Android supported 状态。
- 初始化 deadline 当前在 production factory 返回后判定，能产生稳定超时失败码但不能强制中止永久阻塞的 FFI 调用；真机任务必须仍由外层进程 watchdog 约束 one-shot runner。
- fault fixture 已有独立 never-promote identity/report 通道，但具体 corrupt artifact、load failure、fallback 预期应由后续设备任务提供派生输入并映射到正式 conformance check。
- peak RSS 和 shared-library identity 依赖 Android `/proc` 可读性；读取失败会保持机器可见的缺失/失败结果，不能当作通过。
- bundle 是本地候选，不是发布 artifact；没有上传、安装、签名、合并或清理任务本地缓存。
