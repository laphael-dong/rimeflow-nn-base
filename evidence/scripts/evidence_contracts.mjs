const SHA256 = /^[0-9a-f]{64}$/;
const COMMIT = /^[0-9a-f]{40}$/;
const DAY_MS = 24 * 60 * 60 * 1000;

const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};
const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);
const positive = (value, message) => assert(Number.isFinite(value) && value > 0, message);
const exactKeys = (value, keys, message) => assert(
  value !== null && typeof value === 'object' && !Array.isArray(value)
    && same(Object.keys(value).sort(), [...keys].sort()),
  message,
);

export const PLATFORM_IDS = [
  'macos-arm64-coreml', 'macos-x86_64-coreml', 'ios-arm64-coreml',
  'android-arm64-litert-v2', 'windows-x86_64-windowsml',
  'windows-arm64-windowsml', 'linux-x86_64-ort-cpu',
  'linux-x86_64-ort-accelerated', 'linux-arm64-ort',
  'harmonyos-arm64-mindspore', 'web-wasm-reference',
  'web-webgpu-observation',
];

export const OPERATOR_OBJECTS = [
  ['model', 'models/yolov8n.onnx', '100644', 'blob', '22f19afe710dfa942b3e644c4e5a7ac5c42ac403', 12851098, '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad'],
  ['fixture-manifest', 'evidence/fixtures/manifest.json', '100644', 'blob', 'c1c1e02d97453c082c4fdba49faf3fda901c5efb', 9945, '430c391b9cab0c36b2bde93964e8f6afcd66b0dcb05b055d2728c47679a38acc'],
  ['single-target', 'evidence/fixtures/images/single-target.ppm', '100644', 'blob', '87b410037b98793c76b49eac002d16008634156c', 2073615, '607e88462bd0d80db53b313c12269dff69c608637a582c7a5d2158404b8d4a18'],
  ['web-reference', 'evidence/golden/web-reference.json', '100644', 'blob', '333f5fcf70eafa97abc5967d2c0fb65e9c4eadeb', 29752, '114d7ab29afa5bf9b6246dfd6f1cff37ee3d4fc8212b355d2fb3ac9122f9ce68'],
  ['conversion-summary', 'evidence/conversions/conversion-spikes.json', '100644', 'blob', 'b8ec6d28f6cd834ee7aeea0be1c94c49cf4ada6a', 44390, '55259e0d3fab6e602d7d919fea83b39e0caee864d05b26eb62e68648be03c31b'],
];

export const OPERATOR_PRODUCTION_SOURCES = [
  ['cargo-config', '.cargo/config.toml', '100644', 'blob', '49420ecc39ed1699b6109032bde39856facf0010', 406, '2b2254a04f96694f9cd1e19836f5857b6b86697a1642d842f81490ee45c58384'],
  ['operator-cargo-manifest', 'Cargo.toml', '100644', 'blob', 'cf832b9e74ff316297e5b8e8b2a4445a66c8859c', 2306, 'e433b568ffc864989515735e1126c1f807e5ec8b21fbfa1b31764e18ff55bb4a'],
  ['raw-golden-lockfile', 'evidence/tooling/raw-golden/Cargo.lock', '100644', 'blob', '06c9be3b4e6173736804cde982a3985bc971d8e5', 2855, '560c0ef10865895cd739c0971566564e727b83014ff4f1da3adbe561fa5581e6'],
  ['raw-golden-manifest', 'evidence/tooling/raw-golden/Cargo.toml', '100644', 'blob', '0d2e1bc9e3fc4d5ac5f5cf6a01cbf87868799fe4', 165, '8027af80bec76f757253743a1ef1414e31ce449ab1c8fa5391223b33a2378b0f'],
  ['raw-golden-library', 'evidence/tooling/raw-golden/src/lib.rs', '100644', 'blob', 'f02c07e8a36171ec298fac615457b242cc832653', 4428, '90b7a373ee51547ef5543bd61f37a39c36471d70a0afc7e91917566711933e31'],
  ['raw-golden-runner', 'evidence/tooling/raw-golden/src/main.rs', '100644', 'blob', '4eefa78a1bcbf333d5deb31b9e7de05f4980a0d4', 1770, '37b695ddc00a9573e41179eb27da08fed90f9c88cb784abd51c9bc69b5a4aa73'],
  ['production-postprocess', 'src/postprocess.rs', '100644', 'blob', '98d51d5cebef6057e4f471a8713d69394f62ba86', 2910, '2a07f30ef59aa5d730cc0593adf7e69bc7351f285bdc10a2f7199f4e05d8ff4b'],
];

const tuple = (object) => [object.id, object.path, object.mode, object.type, object.blob, object.bytes, object.sha256];

export function validatePublicationObjects(objects, expected, label) {
  assert(Array.isArray(objects) && objects.length === expected.length, `${label} object set length`);
  assert(new Set(objects.map((item) => item.id)).size === objects.length, `${label} duplicate id`);
  assert(new Set(objects.map((item) => item.path)).size === objects.length, `${label} duplicate path`);
  for (const object of objects) {
    exactKeys(object, ['id', 'path', 'mode', 'type', 'blob', 'bytes', 'sha256'], `${label} exact fields`);
    assert(!object.path.startsWith('/') && !object.path.split('/').includes('..') && !object.path.includes('\\'), `${label} path traversal`);
    assert(object.mode === '100644' && object.type === 'blob', `${label} mode/type`);
  }
  assert(same(objects.map(tuple), expected), `${label} exact tuple set/order`);
}

export function validatePublication(publication) {
  assert(publication.schemaVersion === 1, 'publication schema');
  const { measurementIdentity, historicalReplay, operatorInputPublication, basePublicationState } = publication;
  assert(measurementIdentity.baseHistoricalSourceCommit === '30791ea331532b5b3f7d627cea37e3736765840c', 'historical base identity');
  assert(measurementIdentity.oldEvidenceHead === '852e4330eda982957df0e2bb3b32d6d4934ca01e', 'old evidence identity');
  assert(historicalReplay.head === '2413b775aaba45f81cec4e5a5cb9c24daa1e7ce0' && historicalReplay.isCurrentPublication === false, 'historical replay identity');
  assert(operatorInputPublication.repository === 'https://github.com/laphael-dong/rimeflow-nn-validation.git', 'operator repository');
  assert(operatorInputPublication.ref === 'refs/heads/feature/rimeflow-backend-contract-task1-aggregate', 'operator ref');
  assert(operatorInputPublication.commit === 'c90d3957fbbd04b3f0b29eff7bc873b70eed4400', 'operator commit');
  assert(operatorInputPublication.tree === '341d8b00fb5d4d9afeac856418950c1faa408b2e', 'operator tree');
  validatePublicationObjects(operatorInputPublication.objects, OPERATOR_OBJECTS, 'operator publication');
  validatePublicationObjects(operatorInputPublication.productionSources, OPERATOR_PRODUCTION_SOURCES, 'operator production source');
  assert(basePublicationState.state === 'awaiting-push' && basePublicationState.remoteVerified === false, 'base publication must await push');
  assert(!('commit' in basePublicationState), 'base publication must not self-reference a commit');
  assert(publication.requiredCiState === 'not-established-until-task-3', 'formal CI claim');
}

export function validatePlatformMatrix(matrix, conversionSha256) {
  assert(matrix.schemaVersion === 1 && matrix.frozenBeforeAdapterResults === true, 'matrix schema/freeze');
  const ids = matrix.platforms.map((platform) => platform.id);
  assert(same(ids, PLATFORM_IDS) && new Set(ids).size === ids.length, 'platform set must be exact and ordered');
  assert(Object.keys(matrix.governance).length === 4, 'platform governance fields');
  for (const platform of matrix.platforms) {
    assert(['blocked', 'build-verified'].includes(platform.state), `unsupported platform claim: ${platform.id}`);
    assert(platform.requiredCiState === 'not-established-until-task-3', `CI state: ${platform.id}`);
    assert(platform.operatorConversionEvidence.sha256 === conversionSha256, `operator conversion digest: ${platform.id}`);
    assert(Number.isFinite(platform.timeoutsMs.nativeInitialization) && platform.timeoutsMs.nativeInitialization >= 0, `native timeout: ${platform.id}`);
    assert(Number.isFinite(platform.timeoutsMs.webInitialization) && platform.timeoutsMs.webInitialization >= 0, `web timeout: ${platform.id}`);
  }
}

const REQUIRED_BACKEND_METRICS = ['initializationMs', 'coldInferenceMs', 'warmInferenceMs', 'peakProcessRssBytes'];
const EXTERNAL_PRODUCT_METRICS = ['backendRuntimeArtifactBytes', 'finalPackageBytes', 'finalPackageGrowthBytes', 'finalPackageGrowthRatio', 'rollbackPackage'];
const APPROVAL_FIELDS = ['approver', 'submitter', 'platform', 'metric', 'candidateCommit', 'modelSha256', 'artifactDigest', 'runtimeName', 'runtimeVersion', 'observed', 'threshold', 'reason', 'createdAt', 'expiresAt'];
const EXTERNAL_PRODUCT_POLICY = 'Base 不得用 npm 目录、ORT archive、benchmark binary 或 raw tensor 代替产品指标，也不得宣称这些指标通过。';
const CANDIDATE_ADAPTER_STATE = {
  state: 'not-measured',
  comparisonPublished: false,
  superiorityClaimPublished: false,
  reason: 'candidate adapter 尚未实现；这里只冻结比较规则。',
};
const FROZEN_THRESHOLDS = {
  nativeInitialization: { combination: 'all', relativeToWebMax: 2, absoluteMaxMs: 30000 },
  coldInferenceP95: { combination: 'all', relativeToLegacyOrtMax: 1.25, absoluteMaxMs: 2000 },
  warmInferenceP95: { combination: 'all', relativeToLegacyOrtMax: 1.15, absoluteMaxMs: 1000 },
  peakProcessRss: { combination: 'all', metricKind: 'process-rss-peak', unit: 'bytes', relativeToLegacyOrtMax: 1.25, absoluteMaxBytes: 1073741824 },
};

export function validateThresholds(thresholds, approvals = thresholds.approvalRule?.currentApprovals, now = new Date(), approvalScope = null) {
  exactKeys(thresholds, ['schemaVersion', 'frozenBeforeCandidateAdapterMeasurements', 'frozenAgainst', 'hardGates', 'latencyAndMemoryThresholds', 'externalProductMetrics', 'approvalRule', 'candidateAdapters'], 'threshold top-level fields');
  assert(thresholds.schemaVersion === 3 && thresholds.frozenBeforeCandidateAdapterMeasurements === true, 'threshold schema/freeze');
  assert(same(thresholds.frozenAgainst, {
    baseCommit: '30791ea331532b5b3f7d627cea37e3736765840c',
    operatorCommit: 'eacbcf00dfc2fba941b494e2955e87fffd707382',
    modelSha256: '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad',
  }), 'threshold frozen identity');
  const gates = thresholds.hardGates;
  exactKeys(gates, ['correctnessBeforePerformance', 'thresholdCombination', 'requiredBackendMetrics', 'memoryMetric', 'missingMetricPolicy', 'nonFiniteMetricPolicy', 'mixedDeviceComparisonPolicy', 'mixedExecutionProviderPolicy', 'mixedDigestPolicy'], 'threshold hard gate fields');
  assert(gates.correctnessBeforePerformance === true && gates.thresholdCombination === 'all', 'threshold hard gates');
  assert(same(gates.requiredBackendMetrics, REQUIRED_BACKEND_METRICS), 'backend metrics');
  assert(same(gates.memoryMetric, { name: 'peakProcessRssBytes', metricKind: 'process-rss-peak', unit: 'bytes' }), 'memory metric contract');
  for (const policy of ['missingMetricPolicy', 'nonFiniteMetricPolicy', 'mixedDeviceComparisonPolicy', 'mixedExecutionProviderPolicy', 'mixedDigestPolicy']) assert(gates[policy] === 'fail', `threshold policy ${policy}`);
  assert(same(thresholds.latencyAndMemoryThresholds, FROZEN_THRESHOLDS), 'frozen latency/memory threshold contract');
  for (const threshold of Object.values(FROZEN_THRESHOLDS)) for (const [key, value] of Object.entries(threshold)) if (typeof value === 'number') positive(value, `invalid threshold ${key}`);
  assert(!JSON.stringify(thresholds).includes('peakMemoryBytes'), 'ambiguous peakMemoryBytes alias');
  assert(same(thresholds.externalProductMetrics, {
    owner: 'RimeCut',
    evaluationState: 'not-evaluated-by-base',
    metrics: EXTERNAL_PRODUCT_METRICS,
    policy: EXTERNAL_PRODUCT_POLICY,
  }), 'external product metric contract');
  assert(same(thresholds.candidateAdapters, CANDIDATE_ADAPTER_STATE), 'candidate claim');
  const rule = thresholds.approvalRule;
  exactKeys(rule, ['combination', 'allowedOnlyWhenBaselineCannotRunOrThresholdExceededWithAcceptedTradeoff', 'requiredFields', 'maximumLifetimeDays', 'selfApprovalPolicy', 'missingApprovalPolicy', 'expiredApprovalPolicy', 'crossScopeReusePolicy', 'currentApprovals'], 'approval rule fields');
  assert(rule.combination === 'all' && same(rule.requiredFields, APPROVAL_FIELDS) && rule.maximumLifetimeDays === 90, 'approval rule fields/lifetime');
  assert(rule.allowedOnlyWhenBaselineCannotRunOrThresholdExceededWithAcceptedTradeoff === true, 'approval eligibility policy');
  assert(rule.selfApprovalPolicy === 'fail' && rule.missingApprovalPolicy === 'fail' && rule.expiredApprovalPolicy === 'fail' && rule.crossScopeReusePolicy === 'fail', 'approval failure policies');
  assert(Array.isArray(approvals), 'approval collection');
  if (approvals === rule.currentApprovals) assert(approvals.length === 0, 'tracked approvals must remain empty until a candidate exception request exists');
  for (const approval of approvals) validateApproval(rule, approval, now, approvalScope);
}

export function validateApproval(rule, approval, now, scope = null) {
  for (const field of rule.requiredFields) assert(approval[field] !== undefined && approval[field] !== null && approval[field] !== '', `approval missing ${field}`);
  exactKeys(approval, rule.requiredFields, 'approval exact fields');
  assert(approval.approver !== approval.submitter, 'approval self-approval');
  assert(COMMIT.test(approval.candidateCommit) && SHA256.test(approval.modelSha256) && SHA256.test(approval.artifactDigest), 'approval digest binding');
  assert(Number.isFinite(approval.observed) && approval.observed >= 0 && Number.isFinite(approval.threshold) && approval.threshold >= 0, 'approval numeric values');
  const created = new Date(approval.createdAt);
  const expires = new Date(approval.expiresAt);
  assert(Number.isFinite(created.getTime()) && Number.isFinite(expires.getTime()), 'approval timestamps');
  assert(created <= now, 'approval created in the future');
  assert(expires > created && expires - created <= rule.maximumLifetimeDays * DAY_MS, 'approval lifetime');
  assert(expires >= now, 'approval expired');
  if (scope) for (const field of ['platform', 'metric', 'candidateCommit', 'modelSha256', 'artifactDigest', 'runtimeName', 'runtimeVersion']) assert(approval[field] === scope[field], `approval scope mismatch: ${field}`);
}

export function validatePerformanceContract(performance, publication) {
  assert(!JSON.stringify(performance).includes('peakMemoryBytes'), 'ambiguous peakMemoryBytes alias');
  assert(performance.measurementIdentity.oldEvidenceHead === publication.measurementIdentity.oldEvidenceHead, 'measurement identity drift');
  assert(performance.operatorInputPublication.commit === publication.operatorInputPublication.commit, 'operator publication drift');
  assert(performance.basePublicationState.state === 'awaiting-push', 'base publication state');
  for (const backend of [performance.webWasm, performance.legacyNativeOrt]) {
    const metrics = backend.metrics;
    for (const value of [metrics.initializationMs, metrics.coldInferenceMs, metrics.warmInferenceMs.p50, metrics.warmInferenceMs.p95, metrics.peakProcessRssBytes]) positive(value, 'non-finite, zero, or negative backend metric');
    assert(metrics.warmInferenceMs.samples.length === 30 && metrics.warmInferenceMs.samples.every((value) => Number.isFinite(value) && value > 0), 'invalid warm inference samples');
  }
  positive(performance.legacyNativeOrt.metrics.wgpuInitializationMs, 'invalid wgpu initialization metric');
  assert(performance.webWasm.executionProvider === 'wasm' && performance.webWasm.threads === 1, 'Web must be single-thread WASM');
  assert(performance.legacyNativeOrt.runtime.resolvedExecutionProvider === 'Cpu', 'Native ORT provider');
  assert(performance.legacyNativeOrt.adapter.classification === 'wgpu-preprocess-adapter-not-ort-execution-provider', 'wgpu/ORT provider boundary');
  assert(performance.legacyNativeOrt.metrics.initializationExcludes === 'wgpuInitializationMs', 'initialization lifecycle boundary');
  const providers = performance.comparability.providerIdentity;
  assert(performance.comparability.sameDevice === true && performance.comparability.crossBackendProviderEqual === false, 'cross-backend provider semantics');
  assert(same(providers.web, { provider: 'wasm', roundProviders: ['wasm', 'wasm'], stableAcrossRounds: true }), 'Web provider round identity');
  assert(same(providers.native, { provider: 'Cpu', roundProviders: ['Cpu', 'Cpu'], stableAcrossRounds: true }), 'Native provider round identity');
  assert(performance.comparability.sameModelDigest === true && performance.comparability.sameFixtureDigest === true && performance.comparability.sameInputDigest === true, 'mixed digest comparison');
  assert(performance.productMetrics.owner === 'RimeCut' && performance.productMetrics.evaluationState === 'not-evaluated-by-base', 'product ownership');
  for (const metric of Object.values(performance.packageSizeMetrics)) assert(metric.state === 'blocked', 'blocked product metric state');
}

export function validateRawTensorMetadata(output, expectedSha256) {
  assert(output.classification === 'raw-tensor-test-evidence' && output.productPackaging === false, 'raw tensor classification');
  assert(output.artifact.bytes === 2822400 && output.artifact.sha256 === expectedSha256, 'raw tensor bytes/digest');
  assert(same(output.shape, [1, 84, 8400]) && output.dtype === 'float32' && output.endianness === 'little-endian', 'raw tensor format');
  for (const field of ['modelSha256', 'fixtureSha256', 'inputSha256', 'generationMethod', 'licenseSource']) assert(output[field], `raw tensor missing ${field}`);
}

const REPLAY_STEP_IDS = ['platform-matrix', 'linux-performance-capture-round-1', 'linux-performance-capture-round-2', 'performance-baseline', 'product-package-comparison'];
const REPLAY_STEP_ROUNDS = [2, 1, 1, 2, 0];
const HISTORICAL_STEPS_SHA256 = '90181c2efc5b3155572c1d63718e37275f440c0e13d2df2239bccf88a0837b41';
const REPLAY_TOP_LEVEL_KEYS = [
  'schemaVersion', 'repository', 'historicalReplay', 'measurementIdentity',
  'operatorInputPublication', 'basePublicationState', 'runner', 'tools',
  'immutableLogEvidence', 'steps', 'twoRoundPerformanceOutputDigestsEqual',
  'outputs', 'ordinaryReplayMutatesTrackedEvidence', 'ordinaryReplayValidationChain',
  'task1_6ExecutableClosure', 'task1_6BlockedReason', 'task1_7OwnershipReplayComplete',
];
const REPLAY_RUNNER = { id: 'raffael-HP-EliteBook-845-14-inch-G11-Notebook-PC', owner: 'raffael', os: 'linux', architecture: 'x64' };
const REPLAY_TOOLS = {
  node: { command: ['node', '--version'], exitCode: 0, value: 'v24.15.0', stderr: '' },
  cargo: { command: ['cargo', '--version'], exitCode: 0, value: 'cargo 1.97.1 (c980f4866 2026-06-30)', stderr: '' },
  onnxruntimeWeb: 'onnxruntime-web@1.27.0', nativeOrt: '2.0.0-rc.12',
};
export const REPLAY_OUTPUT_TUPLES = [
  ['evidence/schemas/platform-matrix.schema.json', 4798, '3fd8927707a794618f710c50469c83ab2dac8ad82793829ccf0d4552653f92ad'],
  ['evidence/platform/platform-matrix.json', 26058, 'f187d7b0a6828002ad37d476bce98d85c92a445fccb070980ba390e9c1c9be99'],
  ['evidence/platform/runner-inventory.json', 7886, '74efa7db7ebcf5806984c4b2ff731e43d5e664860eb84deef41955f7f47e5524'],
  ['evidence/reports/local-environment.json', 1218, 'cbbe13827e2f3ff06764c138144d3b286e6dc4ceca7f44b2f0e47348551aacce'],
  ['evidence/performance/backend-thresholds.json', 2716, 'ff0399fb9d04e58cc22cf715782eb02b0f56353ec6fd711a890bede84f1c0c3d'],
  ['evidence/performance/linux-x86_64-capture.json', 13864, 'fe68fed790455fb180dd03305f28f1c882c4a38edfb17fe6810584ac8c972058'],
  ['evidence/performance/linux-x86_64-baseline.json', 19618, '00d901ec79785dd8a42f974d228ea90f26c314c7ae79018ebe2ac66db383d4f1'],
  ['evidence/performance/artifacts/linux-x86_64-single-target-web.f32le.bin', 2822400, '37cc9955c15f1c4283fd2ed0d8aa360905357fd03a386db3a526d94dd0cca212'],
  ['evidence/performance/artifacts/linux-x86_64-single-target-native.f32le.bin', 2822400, 'c84b2985786685fe134fe4dbb7c091ecb72d5be7bdbd442aac832b0995013f27'],
  ['evidence/publication/task1-publication.json', 5625, '14cba0a95925ee55d3ded208dbada220ce1a62aa9e6937a74a729777b857ca9d'],
];
const REPLAY_BLOCKED_REASON = 'Candidate adapter、RimeCut 产品指标、正式 CI 与远端 publication 尚未完成；本地历史基线迁移不代表任务 1.6 完成。';

export function validateReplayManifest(replay, publication, sha256) {
  exactKeys(replay, REPLAY_TOP_LEVEL_KEYS, 'replay exact top-level fields');
  assert(replay.schemaVersion === 3 && replay.repository === 'rimeflow-nn-base', 'replay schema/repository');
  assert(same(replay.historicalReplay, publication.historicalReplay) && same(replay.measurementIdentity, publication.measurementIdentity) && same(replay.operatorInputPublication, publication.operatorInputPublication) && same(replay.basePublicationState, publication.basePublicationState), 'replay provenance drift');
  assert(same(replay.runner, REPLAY_RUNNER), 'replay runner identity drift');
  assert(same(replay.tools, REPLAY_TOOLS), 'replay tool identity drift');
  assert(same(replay.immutableLogEvidence, { kind: 'embedded-historical-manifest', path: 'evidence/replay/task1-replay.json', formalCiState: 'not-established-until-task-3' }), 'replay immutable log/CI identity');
  assert(same(replay.steps.map((step) => step.id), REPLAY_STEP_IDS), 'replay exact step set/order');
  assert(sha256(Buffer.from(JSON.stringify(replay.steps))) === HISTORICAL_STEPS_SHA256, 'replay historical step digest');
  for (let index = 0; index < replay.steps.length; index += 1) {
    const step = replay.steps[index];
    assert(step.rounds.length === REPLAY_STEP_ROUNDS[index], `replay round count: ${step.id}`);
    if (step.id === 'product-package-comparison') {
      assert(step.executed === false && typeof step.blockedReason === 'string' && step.blockedReason.length > 0 && step.repeatComparison === null, 'replay blocked step');
      continue;
    }
    assert(step.executed === true && step.blockedReason === null, `replay executed step: ${step.id}`);
    assert(step.repeatComparison.runs === step.rounds.length && step.repeatComparison.allExitCodesZero === true && step.repeatComparison.deterministicOutputDigestsEqual === true, `replay repeat comparison: ${step.id}`);
    for (let roundIndex = 0; roundIndex < step.rounds.length; roundIndex += 1) {
      const round = step.rounds[roundIndex];
      assert(round.run === roundIndex + 1 && round.exitCode === 0 && round.signal === null, `replay round result: ${step.id}`);
      assert(round.repositoryHead === publication.historicalReplay.head && round.runnerId === replay.runner.id, `replay repository/runner identity: ${step.id}`);
      assert(round.worktreeBefore.tracked === '' && round.worktreeBefore.full === '' && round.worktreeAfter.tracked === '' && round.worktreeAfter.full === '', `replay worktree mutation: ${step.id}`);
      assert(Array.isArray(round.inputs) && round.inputs.every((item) => item.exists === true && Number.isInteger(item.bytes) && item.bytes > 0 && SHA256.test(item.sha256)), `replay input digest: ${step.id}`);
      assert(Array.isArray(round.outputs) && round.outputs.every((item) => item.exists === true && Number.isInteger(item.bytes) && item.bytes > 0 && SHA256.test(item.sha256)), `replay output digest: ${step.id}`);
      assert(round.log.storage === 'embedded-in-replay-manifest' && Number.isInteger(round.log.bytes) && round.log.bytes >= 0 && SHA256.test(round.log.sha256), `replay log metadata: ${step.id}`);
      assert(round.log.bytes === Buffer.byteLength(round.log.stdout) + Buffer.byteLength(round.log.stderr), `replay log bytes: ${step.id}`);
      assert(round.log.sha256 === sha256(Buffer.from(`${round.log.stdout}\0${round.log.stderr}`)), `replay log digest: ${step.id}`);
    }
  }
  assert(replay.twoRoundPerformanceOutputDigestsEqual === true, 'replay performance determinism');
  assert(Array.isArray(replay.outputs) && replay.outputs.length === REPLAY_OUTPUT_TUPLES.length, 'replay exact output ledger length');
  assert(new Set(replay.outputs.map((output) => output.path)).size === replay.outputs.length, 'replay duplicate output path');
  for (const output of replay.outputs) {
    exactKeys(output, ['path', 'bytes', 'sha256'], 'replay output exact fields');
    assert(!output.path.startsWith('/') && !output.path.includes('\\') && !output.path.split('/').includes('..'), 'replay output path traversal');
  }
  assert(same(replay.outputs.map((output) => [output.path, output.bytes, output.sha256]), REPLAY_OUTPUT_TUPLES), 'replay exact output tuples/order');
  assert(replay.ordinaryReplayMutatesTrackedEvidence === false && replay.task1_6ExecutableClosure === false && replay.task1_7OwnershipReplayComplete === false, 'replay completion/mutation claim');
  assert(same(replay.ordinaryReplayValidationChain, ['main-validator', 'performance-negative', 'publication-schema-negative', 'replay-negative', 'main-security-negative', 'official-trust-negative']), 'ordinary replay validation chain');
  assert(replay.task1_6BlockedReason === REPLAY_BLOCKED_REASON, 'replay task 1.6 blocked reason drift');
}

export function assertTrackedUnchanged(before, after, beforeDiff, afterDiff) {
  if (JSON.stringify(before) !== JSON.stringify(after) || !beforeDiff.equals(afterDiff)) throw new Error('ordinary replay modified tracked evidence');
}
