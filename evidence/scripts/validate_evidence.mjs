import { createHash } from 'node:crypto';
import { lstat, readFile, realpath } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  PLATFORM_IDS,
  validatePerformanceContract,
  validatePlatformMatrix,
  validatePublication,
  validateRawTensorMetadata,
  validateReplayManifest,
  validateThresholds,
} from './evidence_contracts.mjs';
import { validateFrozenPlatformSchema, validateJsonSchema } from './json_schema_validation.mjs';
import { OFFICIAL_LINUX_X64_TOOLS } from './official_live_trust.mjs';
import { verifyOperatorExport } from './operator_input_export.mjs';
import { validatePerformanceCapture } from './performance_validation.mjs';

const defaultRoot = resolve(import.meta.dirname, '../..');
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);

export async function validateEvidence({
  root = defaultRoot,
  operatorBundleRoot = process.env.RIMEFLOW_OPERATOR_BUNDLE,
  operatorRoot = operatorBundleRoot ? resolve(operatorBundleRoot, 'source') : null,
  operatorGit = OFFICIAL_LINUX_X64_TOOLS.git.path,
  operatorGitEnv = { PATH: '/usr/bin:/bin', GIT_EXEC_PATH: '/usr/lib/git-core', LC_ALL: 'C' },
  productionRunner = null,
  productionRunnerEnv = process.env,
  scratchRoot = null,
} = {}) {
  const readJson = async (base, path) => JSON.parse(await readFile(resolve(base, path), 'utf8'));
  const publication = await readJson(root, 'evidence/publication/task1-publication.json');
  validatePublication(publication);
  if (!operatorBundleRoot || !operatorRoot) throw new Error('ordinary validation requires RIMEFLOW_OPERATOR_BUNDLE from fresh exact export');
  const operatorExport = await verifyOperatorExport(operatorBundleRoot, publication, { git: operatorGit, gitEnv: operatorGitEnv });
  if (operatorExport.source !== resolve(operatorRoot)) throw new Error('operator source must be derived from verified bundle');

  const conversion = publication.operatorInputPublication.objects.find((object) => object.id === 'conversion-summary');
  const matrix = await readJson(root, 'evidence/platform/platform-matrix.json');
  const matrixSchemaBytes = await readFile(resolve(root, 'evidence/schemas/platform-matrix.schema.json'));
  const matrixSchema = JSON.parse(matrixSchemaBytes);
  const matrixSchemaIdentity = validateFrozenPlatformSchema(matrixSchemaBytes, matrixSchema);
  validateJsonSchema(matrixSchema, matrix);
  validatePlatformMatrix(matrix, conversion.sha256);
  if (!sameJson(matrix.platforms.map((platform) => platform.id), PLATFORM_IDS)) throw new Error('platform set drift');
  const runnerInventory = await readJson(root, 'evidence/platform/runner-inventory.json');
  const requiredRunnerTargets = ['macos-arm64', 'macos-x86_64', 'ios-arm64-device', 'android-arm64-device', 'windows-x86_64', 'windows-arm64', 'linux-x86_64-cpu', 'linux-x86_64-accelerated', 'linux-arm64', 'harmonyos-arm64-device'];
  if (!sameJson(runnerInventory.runners.map((runner) => runner.target), requiredRunnerTargets)) throw new Error('runner set drift');
  for (const runner of runnerInventory.runners) {
    if (!['blocked', 'build-verified'].includes(runner.state) || !runner.requiredOwnerRole || !runner.requiredCiJob || !runner.unblockCondition) throw new Error(`runner semantics: ${runner.target}`);
    if (runner.state === 'blocked' && (runner.runnerId !== null || runner.owner !== null || runner.ciAvailability !== false)) throw new Error(`blocked runner identity: ${runner.target}`);
  }

  const environment = await readJson(root, 'evidence/reports/local-environment.json');
  if (environment.schemaVersion !== 2 || environment.classification !== 'historical-measurement-environment' || environment.host.os !== 'linux' || environment.host.arch !== 'x64') throw new Error('environment identity');
  if (environment.toolchains.androidDevice !== 'none' || environment.toolchains.androidNdk.state !== 'not-observed' || environment.toolchains.androidNdk.value !== null) throw new Error('android NDK observation semantics');

  const thresholds = await readJson(root, 'evidence/performance/backend-thresholds.json');
  validateThresholds(thresholds, undefined, new Date('2026-08-13T00:00:00.000Z'));
  const replay = await readJson(root, 'evidence/replay/task1-replay.json');
  validateReplayManifest(replay, publication, digest);
  for (const output of replay.outputs) {
    const path = resolve(root, output.path);
    const canonical = await realpath(path);
    if (!canonical.startsWith(`${resolve(root)}/`)) throw new Error(`tracked evidence path escaped root: ${output.path}`);
    const metadata = await lstat(path);
    if (!metadata.isFile() || metadata.isSymbolicLink()) throw new Error(`tracked evidence is not a regular file: ${output.path}`);
    const bytes = await readFile(path);
    if (bytes.length !== output.bytes || digest(bytes) !== output.sha256) throw new Error(`tracked evidence drift: ${output.path}`);
  }
  const capturePath = 'evidence/performance/linux-x86_64-capture.json';
  const captureBytes = await readFile(resolve(root, capturePath));
  const capture = JSON.parse(captureBytes);
  const performance = await readJson(root, 'evidence/performance/linux-x86_64-baseline.json');
  if (performance.source.measurementCapture.path !== capturePath || performance.source.measurementCapture.sha256 !== digest(captureBytes)) throw new Error('measurement capture digest');
  const expectedPerformance = {
    ...capture,
    measurementIdentity: publication.measurementIdentity,
    operatorInputPublication: publication.operatorInputPublication,
    basePublicationState: publication.basePublicationState,
    source: {
      ...capture.source,
      measurementCapture: performance.source.measurementCapture,
      evidenceHarness: { ...capture.source.evidenceHarness, reportGeneratorSha256: performance.source.evidenceHarness.reportGeneratorSha256 },
    },
  };
  if (!sameJson(performance, expectedPerformance)) throw new Error('performance report does not match frozen capture');
  validatePerformanceContract(performance, publication);
  validateRawTensorMetadata(performance.webWasm.output, publication.measurementIdentity.webRawOutputSha256);
  validateRawTensorMetadata(performance.legacyNativeOrt.output, publication.measurementIdentity.nativeRawOutputSha256);
  const effectiveScratchRoot = scratchRoot ?? resolve(root, '.evidence/task1-linux-baseline/validator-postprocess');
  const effectiveRunnerEnv = productionRunner ? productionRunnerEnv : { ...productionRunnerEnv, CARGO_TARGET_DIR: resolve(effectiveScratchRoot, 'cargo-target') };
  await validatePerformanceCapture(root, operatorRoot, performance, { productionRunner, productionRunnerEnv: effectiveRunnerEnv, scratchRoot: effectiveScratchRoot });
  return {
    ok: true, schemaVersion: 4, platformCount: matrix.platforms.length, supportedCount: 0,
    operatorInputCommit: publication.operatorInputPublication.commit,
    basePublicationState: publication.basePublicationState.state,
    linuxComparisonPassed: true, candidateComparisonPublished: false,
    formalCiState: publication.requiredCiState,
    validationMode: productionRunner ? 'official-live-fresh-runner' : 'ordinary-offline',
    operatorExport: { commit: operatorExport.fetchHead, tree: operatorExport.tree, exportedFileCount: operatorExport.exportedFileCount, verifiedObjectCount: operatorExport.verified.length },
    platformSchema: matrixSchemaIdentity,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  console.log(JSON.stringify(await validateEvidence()));
}
