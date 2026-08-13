# 任务 1 平台与性能基线证据

本目录保存从旧 `rimeflow-onnx-base` 历史证据修复迁入 `rimeflow-nn-base` 的平台矩阵、Legacy Native ORT/Web ONNX 性能基线和 replay harness。它是本地 implementation delivery，不代表 OpenSpec 任务 1.6、1.7、1.8 完成，也不代表 publication closure。

## 身份与 publication

机器可读入口是 `evidence/publication/task1-publication.json`，其中四类身份严格分离：

- `measurementIdentity`：历史测量来源。base source 是 `30791ea331532b5b3f7d627cea37e3736765840c`，旧 evidence range 是 `30791ea..852e433`；这些值不会机械替换为迁移提交。
- `historicalReplay`：历史 capture/replay HEAD `2413b775aaba45f81cec4e5a5cb9c24daa1e7ce0`，明确不是当前 publication。
- `operatorInputPublication`：正式 validation input 固定为 `rimeflow-nn-validation` 的 `c90d3957...`、tree `341d8b...`，并逐项锁定模型、fixture manifest、single-target、Web reference 和 conversion summary 的 path/blob/bytes/SHA-256。
- `basePublicationState`：当前为 `awaiting-push`。提交内容不自引用未知最终 SHA；后续 push/ledger 任务必须从运行时 HEAD 推导并验证远端对象。

`node evidence/scripts/verify_operator_input_live.mjs` 是 official live verification。它在单一受控进程内创建全新的临时 bare repository，以独立 Linux PGID 执行 `git fetch --no-tags --depth=1 <repository> <ref>`，验证 `FETCH_HEAD`、tree，导出并逐 blob 重验完整 regular-file source tree；其中 5 个 publication input 和 7 个生产 Rust 后处理关键对象还会精确验证 mode/type/blob/bytes/SHA-256。timeout、失败、signal、输出异常或leader提前退出但后代仍存活时，只向该任务PGID依次发送SIGTERM和有界宽限期后的SIGKILL，等待直接child close并证明PGID无成员后才删除partial目录；不得按进程名清理或影响其他Git任务。成功receipt同样绑定leader PID/PGID、exit/close和零残留状态。随后把 source tree 设为只读并逐项验证权限，使用 fresh `CARGO_HOME`、fresh target 和 `cargo build --locked --jobs 1 --release` 构建 raw-golden runner，只执行 fresh target 中经过regular-file、owner/mode、host、bytes和SHA-256验证的runner，并在同一临时目录生命周期内运行主evidence validation。网络、工具、权限、构建、identity或断言失败都会以 `live verification failed/unavailable` 和非零退出码报告，绝不降级成ordinary/offline success。

official-live 支持面固定为当前 Linux x64 runner。Node、Git、Git HTTPS launcher/target、Cargo、rustc、GCC linker、collect2、ld、assembler、ar 和 ranlib 均按 absolute/canonical path、owner/mode、parent directory、version 与 SHA-256 验证，不通过调用者 `PATH` 解析。合法的 `/usr/lib/git-core/git-remote-https -> git-remote-http` 被建模为 root-owned exact symlink，并单独验证 resolved target。调用者的 `PATH`、`RUSTC`、wrapper、Rust flags、Cargo rustc/target/home 和 rustup home/toolchain不会继承；只有 proxy/CA 网络变量按白名单转发。

直接运行 `validate_evidence.mjs` 是 ordinary offline validation，必须使用 `prepare_operator_input.mjs` 生成且重新验证的 fresh bundle，并在独立 scratch target 中执行后处理；它不得冒充 official live publication verification。

## 平台与 CI 边界

平台集合由 validator 精确锁定为 12 项，拒绝缺失、重复、未知目标和 `supported` 声明。当前只有 `blocked` 与 `build-verified`。`requiredCiState` 固定为 `not-established-until-task-3`；`requiredCiJob` 是未来任务的合同名称，不是已存在或已通过的 CI 证据。

`local-environment.json` 是历史测量环境，不是当前主机探测结果。Android NDK 在该 Linux capture 中是 `not-observed`/`null`；Android 平台合同要求由未来受信目标 runner 证明 NDK r27c，不能把未观察到的环境伪造成 `r30-beta1`。

## 性能与所有权

正式可比较范围只是同一 Linux x86_64 主机的 backend microbenchmark：

- Web 是 `onnxruntime-web@1.27.0`、WASM、单线程。
- Native 是 `rimeflow-onnx-base::NativeOrtBackend`，resolved ORT execution provider 是 CPU。
- Web 与 Native 的 provider 本来就不同，`crossBackendProviderEqual=false`。`mixedExecutionProviderPolicy=fail` 表示 Web 的两轮样本都必须保持 WASM、Native 的两轮样本都必须保持 CPU，不能在同一 backend 的 rounds 中混入其他 provider；它不要求跨 backend provider 相等。
- 报告中的 Vulkan/RADV 是 wgpu 预处理 adapter，不是 Vulkan ORT execution provider。
- Native `initializationMs` 不包含单独记录的 `wgpuInitializationMs`；不声称 Web 与 Native 初始化生命周期完全等价。
- 内存指标只允许 `peakProcessRssBytes`，阈值 `metricKind=process-rss-peak`。旧别名 `peakMemoryBytes` 会被拒绝。

base 只拥有初始化、冷/热推理和进程 RSS 峰值。`backendRuntimeArtifactBytes`、`finalPackageBytes`、`finalPackageGrowthBytes`、`finalPackageGrowthRatio` 与 rollback package 由 RimeCut 拥有，base 状态是 `not-evaluated-by-base`。npm 目录、ORT archive、benchmark binary 和 raw tensor 都不能填入这些产品指标。

阈值使用 `combination=all` 和 `missingMetricPolicy=fail`。审批 schema 冻结 approver/submitter、平台、metric、candidate commit、模型与 artifact digest、runtime/version、observed/threshold、reason、createdAt/expiresAt，最长 90 天且禁止自批、future-created、过期和跨 scope 复用。当前没有可验证的 candidate exception/request identity，因此 tracked `currentApprovals` 必须严格为空；未来任务只有在新增并验证真实 request scope 后才能支持非空 tracked approvals，不能由调用者单独提供 scope。

## Raw tensor test evidence

两个 `.f32le.bin` 文件各 2,822,400 字节，shape `[1,84,8400]`、dtype `float32`、little-endian。它们分类为 `raw-tensor-test-evidence`，`productPackaging=false`，并记录模型、fixture、canonical input digest、生成方式和许可来源。validator/负测拒绝截断、延长、NaN、Infinity、SHA、shape、dtype、endianness 漂移和产品打包越权。

## 生成与 replay

以下链路只从锁定输入确定性更新受管 JSON，不重新运行 benchmark：

```bash
node evidence/scripts/generate_platform_evidence.mjs
node evidence/scripts/generate_performance_baseline.mjs
node evidence/scripts/migrate_historical_replay.mjs
```

连续两轮必须得到逐字节相同输出。先用受信工具链进行 fresh fetch 和完整 source export；目标目录必须不存在：

```bash
node evidence/scripts/prepare_operator_input.mjs /absolute/new/operator-bundle
RIMEFLOW_OPERATOR_BUNDLE=/absolute/new/operator-bundle node evidence/scripts/replay_task1.mjs
```

bundle 固定 repository/ref/commit `c90d3957fbbd04b3f0b29eff7bc873b70eed4400`/tree `341d8b00fb5d4d9afeac856418950c1faa408b2e`，包含 fresh bare repo、完整只读 source 和 receipt；每次 ordinary validation/replay 都重新验证 bare `FETCH_HEAD`、tree、完整 path/blob 集合及 12 个关键 tuple。普通 replay 严格验证历史 step 集合、顺序、repository/input/output/log digest、round/exit code、repeatComparison、blocked 状态和两轮确定性，然后按 manifest 固定顺序实际执行主 validator、performance negative、publication/schema negative、strict replay negative、main security negative 和 official trust negative 六条链。它对全部 tracked 文件做执行前后 SHA 映射及 binary diff 比较，任何 mutation 都失败。历史 log/rounds 保持历史身份。`generate_task1_replay_manifest.mjs` 的 record 模式必须显式设置 `RIMEFLOW_RECORD_REPLAY=1`；采集器只有在显式 `RIMEFLOW_RECORD_BASELINE=1` 时才允许覆盖 tracked capture，并使用临时文件原子发布。任务不运行产品 benchmark。

## 验证命令

```bash
node evidence/scripts/prepare_operator_input.mjs /absolute/new/operator-bundle
RIMEFLOW_OPERATOR_BUNDLE=/absolute/new/operator-bundle node evidence/scripts/validate_evidence.mjs
node evidence/scripts/verify_operator_input_live.mjs
RIMEFLOW_OPERATOR_ROOT=/absolute/new/operator-bundle/source node evidence/scripts/test_performance_validator_negative.mjs
node evidence/scripts/test_publication_validator_negative.mjs
node evidence/scripts/test_replay_validator_negative.mjs
RIMEFLOW_OPERATOR_BUNDLE=/absolute/new/operator-bundle node evidence/scripts/test_main_validator_security_negative.mjs
node evidence/scripts/test_official_live_trust_negative.mjs
node evidence/scripts/test_official_live_hostile_e2e.mjs
RIMEFLOW_OPERATOR_BUNDLE=/absolute/new/operator-bundle node evidence/scripts/replay_task1.mjs
```

当前没有 candidate adapter comparison、superiority claim、formal CI success、supported/recommended 状态或 publication verified 声明。
