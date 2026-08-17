import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { validateJsonSchema } from './json_schema_validation.mjs';

const SHA256 = /^[0-9a-f]{64}$/;
const COMMIT = /^[0-9a-f]{40}$/;
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

export function validateAndroidBaseClosure({ closure, validationReport, deviceSummary, conformanceReport, schema, validationReportSha256, deviceManifestSha256, executionSummarySha256 }) {
  validateJsonSchema(schema, closure);
  const identities = closure.identities;
  assert(COMMIT.test(identities.baseCommit) && COMMIT.test(identities.runnerCommit), 'closure Base/runner commit identity');
  assert(identities.baseCommit === '52c85ec737dc4b4ab482f6c2654f914150c4dfae', 'closure runner commit drift');
  assert(identities.validationCommit === 'eef2fabf33b954fd75f7772db4c2175fda03dde7', 'Validation DCO identity drift');
  assert(identities.validationCandidateCommit === 'c7bca1cdc1235bbb294c0e40cdfcfacfdc95a660', 'Validation candidate identity drift');
  assert(SHA256.test(identities.validationReportSha256) && identities.validationReportSha256 === validationReportSha256, 'Validation report SHA-256 drift');
  assert(SHA256.test(identities.deviceManifestSha256) && identities.deviceManifestSha256 === deviceManifestSha256, 'device manifest SHA-256 drift');
  assert(SHA256.test(identities.executionSummarySha256) && identities.executionSummarySha256 === executionSummarySha256, 'execution summary SHA-256 drift');

  const candidate = validationReport.candidate;
  assert(candidate.runnerBundleId === identities.bundleId && candidate.runnerCommit === identities.runnerCommit, 'Validation candidate runner identity drift');
  assert(candidate.tfliteSha256 === identities.tfliteSha256 && candidate.validationCommit === identities.validationCandidateCommit, 'Validation candidate artifact identity drift');
  assert(validationReport.status?.finalPlatformClose === false && validationReport.status?.supported === false && validationReport.status?.performancePassed === false, 'Validation report overclaims platform close');
  assert(validationReport.performance?.outcome === 'exception-recorded-no-comparative-pass' && validationReport.performance.performancePassed === false && validationReport.performance.samePhoneBaselineAvailable === false, 'Validation performance boundary drift');

  const device = deviceSummary.deviceIdentity;
  assert(JSON.stringify(closure.device) === JSON.stringify({
    serial: device.serial,
    manufacturer: device.manufacturer,
    model: device.model,
    android: device.android,
    apiLevel: device.apiLevel,
    abi: device.abi,
    fingerprint: device.fingerprint,
  }), 'device identity drift');
  const accepted = deviceSummary.acceptedBundleIdentity;
  for (const [key, identityKey] of [
    ['bundleId', 'bundleId'],
    ['bundleManifestSha256', 'bundleManifestSha256'],
    ['runnerSha256', 'runnerSha256'],
    ['runtimeSha256', 'runtimeSha256'],
    ['tfliteSha256', 'tfliteSha256'],
    ['generatedModelManifestSha256', 'generatedModelManifestSha256'],
    ['historicalArtifactManifestSha256', 'historicalArtifactManifestSha256'],
  ]) assert(accepted[key] === identities[identityKey], `accepted bundle ${key} drift`);
  assert(deviceSummary.ioContract.status === 'raw-io-contract-observation' && deviceSummary.ioContract.signatureBindingNamesSeparateFromRuntimeNames === true, 'device I/O boundary drift');
  assert(JSON.stringify(closure.ioContract.input) === JSON.stringify({
    role: 'image', signatureBindingName: 'args_0', runtimeName: 'serving_default_args_0', runtimeIndex: 0,
    shape: [1, 3, 640, 640], dtype: 'f32', layout: 'NCHW',
  }), 'closure input binding/runtime tensor identity drift');
  assert(JSON.stringify(closure.ioContract.output) === JSON.stringify({
    role: 'detections', signatureBindingName: 'output_0', runtimeName: 'serving_default_output_0_output', runtimeIndex: 0,
    shape: [1, 84, 8400], dtype: 'f32', layout: 'NCHW',
  }), 'closure output binding/runtime tensor identity drift');
  assert(deviceSummary.fixtures.acceptedFixtureCount === 5 && deviceSummary.fixtures.runsPerFixture === 2 && deviceSummary.fixtures.outputCount === 10 && deviceSummary.fixtures.deterministic === true, 'device fixture evidence drift');
  assert(deviceSummary.faults.acceptedBytesUnchanged === true && deviceSummary.faults.derivedInputsNeverPromote === true, 'fault boundary drift');
  assert(deviceSummary.packageLoad.status === 'passed' && deviceSummary.packageLoad.runtimeShaMatches === true && deviceSummary.packageLoad.accelerator === 'CPU', 'package-load evidence drift');
  assert(deviceSummary.comparator.meaningfulSamePhoneComparatorAvailable === false && deviceSummary.comparator.absoluteMetricException.notPerformanceApproval === true, 'device comparator exception drift');
  assert(deviceSummary.comparatorException.expiry === 'independent Validation report completion', 'device exception expiry drift');

  const golden = validationReport.goldenComparison;
  assert(golden.summary.passed === true && golden.summary.deterministic === true && golden.summary.fixtureCount === 5, 'Validation golden summary drift');
  assert(golden.summary.rawMismatchCount === 0 && golden.summary.anchorMismatchCount === 0 && golden.summary.classMismatchCount === 0 && golden.summary.countMismatchCount === 0, 'Validation golden mismatch drift');
  assert(closure.gates.performance.performancePassed === false && closure.performancePassed === false && closure.supported === false && closure.finalPlatformClose === false, 'closure must remain unsupported');

  assert(conformanceReport.runner.kind === 'real-target' && conformanceReport.runner.runnerId === closure.device.serial, 'conformance runner identity drift');
  assert(conformanceReport.selection.kind === 'ready', 'conformance selection must be ready');
  const checks = new Map(conformanceReport.checks.map((check) => [check.kind, check]));
  for (const kind of ['manifest_io', 'initialization_timeout', 'smoke_inference', 'golden_output', 'fault_injection', 'diagnostics', 'package_load']) assert(checks.get(kind)?.status === 'passed', `conformance gate ${kind} must pass`);
  assert(checks.get('performance')?.status === 'blocked', 'conformance performance gate must remain blocked');
  return { ok: true, status: closure.status, supported: closure.supported, finalPlatformClose: closure.finalPlatformClose };
}

const parseArgs = (argv) => {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    if (!argv[index].startsWith('--') || !argv[index + 1]) throw new Error(`expected --name value, got ${argv[index] ?? '<end>'}`);
    args[argv[index].slice(2)] = argv[++index];
  }
  return args;
};

if (process.argv[1]?.endsWith('validate_android_base_closure.mjs')) {
  const args = parseArgs(process.argv.slice(2));
  const readJson = async (path) => JSON.parse(await readFile(resolve(path), 'utf8'));
  const [closureBytes, schemaBytes, validationBytes, manifestBytes, summaryBytes, conformanceBytes] = await Promise.all([
    readFile(resolve(args.closure)),
    readFile(resolve(args.schema)),
    readFile(resolve(args.validationReport)),
    readFile(resolve(args.deviceManifest)),
    readFile(resolve(args.deviceSummary)),
    readFile(resolve(args.conformance)),
  ]);
  const result = validateAndroidBaseClosure({
    closure: JSON.parse(closureBytes),
    schema: JSON.parse(schemaBytes),
    validationReport: JSON.parse(validationBytes),
    deviceSummary: JSON.parse(summaryBytes),
    conformanceReport: JSON.parse(conformanceBytes),
    validationReportSha256: sha256(validationBytes),
    deviceManifestSha256: sha256(manifestBytes),
    executionSummarySha256: sha256(summaryBytes),
  });
  console.log(JSON.stringify(result));
}
