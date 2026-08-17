# RFB-ANDROID-RUNNER-01 repair 实现报告

## 结论与边界

本 repair 在现有 `rfb-android-runner-01` 分支修正 LiteRT 2.1.6 signature binding 与 subgraph tensor identity 混用的问题，并生成新的 Android arm64-v8a/API 26+/CPU-only bundle。pre-repair commit 为 `38457d69832aec13fc2c627fe1af7534a4861bac`；delivery commit 是包含本报告的 DCO commit，精确远端 SHA 由 Dispatch feedback 和 PR #1 readback 记录，以避免在提交内容中建立自摘要。

本任务没有运行 ADB，没有向 RFCX90C568W 或任何手机写入、安装或执行内容，没有更换 TFLite、Validation metadata、LiteRT/NDK/Rust 版本、manifest 逻辑 I/O、shape、dtype、layout 或 tolerance，也没有修改 supported 状态、RimeCut、OpenSpec ledger、历史 blocked report 或已失败 bundle。结果仅代表 repaired build candidate，可供另一个独立、零重试 device acceptance Task 使用，不构成设备验收。

## 首失败与根因

已读取 device Task `task_05ac0a682fd0` / Dispatch `ctx_aa84276ef276` 的 report 和 execution summary。身份、部署与 digest parity 已通过；首个真实失败是：

```text
stage: IoDiscovery
code: litert_io_contract_mismatch
message: LiteRT input args_0 discovery failed: Error: SubgraphInputTensorByNameNotFound
```

冻结 bundle manifest 与 Validation metadata 的实际输入 tensor 名为 `serving_default_args_0`，signature position 为 0；实际输出 tensor 名为 `serving_default_output_0_output`。设备上 `Signature::input_names()` 返回的是 binding name `args_0`。旧 adapter discovery 与 vendored `CompiledModel::create_*_tensor_buffers` 都把 binding name 误传给 `Subgraph::*_tensor_by_name`，因此在模型编译成功后、buffer 创建前无法绑定输入。

LiteRT 2.1.6 `litert_model.h` 明确提供：

```c
LiteRtGetSignatureInputTensorByIndex(signature, input_idx, &tensor);
LiteRtGetSignatureOutputTensorByIndex(signature, output_idx, &tensor);
```

同时 compiled-model buffer requirements 也以 `signature_index + input/output_index` 为身份。由此采用唯一映射：signature binding name 只作为 binding identity/诊断；signature position 通过上述 C API 得到实际 tensor；实际 tensor name、position、runtime dtype/shape 再与 manifest 的 logical role 做 AND 匹配。arity、空/重复 binding、缺失/歧义 tensor、动态/无效 shape 以及 dtype/shape drift 均保持硬失败。

## 精确源修复

- `src/backend/litert_v2/android.rs`：移除 subgraph name lookup；input/output 均按 signature index 获取 tensor；从 runtime tensor 读取 name、ranked shape 与 dtype；严格验证 binding arity/唯一性和 manifest identity。
- `src/backend/litert_v2.rs`：descriptor 与 diagnostics binding 新增向后兼容的可选 `signatureBindingName`；`runtimeName` 继续表示实际 tensor name。
- vendored `model.rs`：分别包装官方 C API `LiteRtGetSignatureInputTensorByIndex` 和 `LiteRtGetSignatureOutputTensorByIndex`。
- vendored `compiled_model.rs`：buffer requirements 与 tensor 都使用相同 signature position；仍完整遍历并验证 binding-name iterator 的错误。
- vendored `error.rs`：只新增两个准确的 error causes。
- `tests/litert_v2_adapter.rs`：回归同时断言 input/output binding name 与实际 runtime name 可不同；production source audit 断言 adapter 与 buffer path 不再使用 subgraph name lookup，并覆盖两个官方 C API。该测试在旧实现上会因缺少 accessor/仍存在 name lookup 而失败；Android target compile/link 进一步验证真实 bindgen 符号，不仅依赖 fake runtime。

production 路径保持 `AndroidLiteRtV2Factory -> CompiledModel -> RuntimeBackend::infer`，CPU-only、one-shot factory、初始化阶段码、smoke 前 fallback 和 Ready 后无 fallback 语义均未改变。

## Official-vendor delta

官方 crate 文件来自 Cargo registry `google-ai-edge-litert-0.1.3.crate`，SHA-256 为 `fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8`。pre-repair vendor 与官方的既有差异只在 task-local pinned build path、Cargo metadata 和 Android byte-buffer compatibility；本 repair 对官方 runtime binding 源的新增 delta 仅为：

- `src/model.rs`：40 行，两个薄 C API wrapper。
- `src/compiled_model.rs`：input/output 各移除 subgraph lookup，并改用对应 signature-index accessor；总计 4 additions/4 deletions。
- `src/error.rs`：2 行 error-cause variants。

没有改变 crate 版本、LiteRT runtime ABI、下载 URL/SHA、Cargo dependency graph 或 `[patch.crates-io]` 范围。

## 固定输入、工具与 bundle identity

| 项目 | 版本 / SHA-256 |
| --- | --- |
| Rust | `rustc 1.97.1 (8bab26f4f 2026-07-14)` |
| Cargo | `cargo 1.97.1 (c980f4866 2026-06-30)` |
| CMake | `3.28.3` |
| NDK r27c | `27.2.12479018`; zip `59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc` |
| LiteRT C++ SDK 2.1.6 | `2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8` |
| LiteRT Android arm64 runtime 2.1.6 | `35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369` |
| Rust binding | `0.1.3`; official crate `fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8` |
| frozen TFLite | `794e17d9a2795084787e5125bcfada6cb501c6afc4f708e7e58e96fd8dc84be1` |
| frozen Validation conversion metadata | `f1f95bac2006cd02e364123ab9e7556cc331e6d20f49006c50fcebd15d0c4881` |

新 bundle 保存在独立目录 `.android-litert/bundles/sha256-0c309e3f63aa69920a1ac55a42b0a5211771671bf7df0fa180205550ce535866`；旧失败 bundle `sha256-dfd7ffebf87f7c120bb0802cb764f2de0d3f7c1c0bbbaf5119081e64a6e6b807` 保留且未覆盖。

- bundle ID：`sha256-0c309e3f63aa69920a1ac55a42b0a5211771671bf7df0fa180205550ce535866`
- bundle manifest SHA-256：`62fa77644a12d42391a20a80523fdf81825cdd31737edebbd0d5f1c012c3d9b5`
- runner SHA-256：`627206702b9ed1260b3f3d7b0bb24c543c7fa4d4f8978ae587a10f173f8d8033`
- runtime SHA-256：`35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369`
- generated model manifest SHA-256：`1b059b63ca70f2bb353efa718aed75b428b4361056149fdfc5012738e6ce6e9b`
- build provenance SHA-256：`4f313e4415266e9a165a5216c2deb58311c1129f3350baa0938eaeb2fd6554d2`

`package_bundle.mjs --verify` 对新 bundle 返回 `verified: true`。runner 是 64-bit little-endian AArch64 PIE，interpreter `/system/bin/linker64`；NEEDED 为 `libdl.so`、`libLiteRt.so`、`libc.so`，无 RPATH/RUNPATH。bundle manifest 保持 API 26、CPU-only、5 个 digest-addressed frozen golden inputs、2 次 golden capture、5 次 warmup、30 次 warm sample、peak RSS 与 package-load hooks。

## 修复周期与验证

首轮 focused host regression 为 6/6 passed。首个 Android target build 失败于 Rust import：新增 `Tensor` wrapper 已在官方 `litert::model` public module 中，但不在 crate root re-export；将 import 收窄为 `litert::model::Tensor` 后，未修改 vendor public surface，完整回归和 NDK r27c release compile/link/package 一次通过。两个允许的 correction cycle 只使用一个。

- `cargo fmt --all -- --check`：passed。
- `cargo test --test litert_v2_adapter`：6 passed，0 failed。
- `cargo test`：全量默认 host tests passed，0 failed；1 个 doc test ignored。
- `bash -n tools/android-litert-runner/build_bundle.sh`：passed。
- `node --check tools/android-litert-runner/package_bundle.mjs`：passed。
- NDK r27c `aarch64-linux-android` release compile/link/package：passed。
- `node tools/android-litert-runner/package_bundle.mjs --verify <new-bundle>`：passed。
- `file` / `readelf -h` / `readelf -d`：AArch64 Android identities passed；无 RPATH/RUNPATH。
- `git diff --check`：passed。

原始构建日志保存在 ignored `.android-litert/repair-cycle-1-build.log`（首失败）和 `.android-litert/repair-cycle-1-correction-build.log`（成功）。这些日志没有上传，task-local cache/output 均保留供管理复核。

## Evidence manifest 与残余风险

失败 device evidence 的 `evidence-manifest.json` 把自身列为 0 bytes / empty SHA，同时又声明 self-exclusion。Base runner 与 bundle helper 中不存在该 device evidence manifest 生成逻辑，因此本 repair 没有跨 ownership 修改 device worktree。下一次 device Task 必须从 `files` 列表完全排除 `evidence-manifest.json`，并只以外部摘要交付 manifest identity；不得再写入误导性的零字节 self-entry。

- 尚无 repaired bundle 的设备执行证据；signature-index C API、target link 与 bundle integrity 已验证，但初始化、真实 output、timing、RSS 和 `/proc/self/maps` package-load 仍必须由新的零重试 device Task 证明。
- output binding mismatch 路径已按相同 C API 和 regression 覆盖，但先前设备执行在 input discovery 终止，尚无该物理设备的 output discovery 观测。
- 初始化 FFI 的进程内不可抢占风险、Android `/proc` 可读性和 fault-never-promote 派生输入要求与原实现报告一致。
- bundle 仍是本地候选；没有上传、签名、安装、release、merge 或 supported-status 变更。
