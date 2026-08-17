import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { validateJsonSchema } from './json_schema_validation.mjs';
import { validateAndroidBaseClosure } from './validate_android_base_closure.mjs';
import { validateRunnerInventory } from './validate_evidence.mjs';

const root = resolve(import.meta.dirname, '../..');
const closure = JSON.parse(await readFile(resolve(root, 'reports/rfb-android-base-closure-01.json'), 'utf8'));
const schema = JSON.parse(await readFile(resolve(root, 'schemas/android-base-closure.schema.json'), 'utf8'));
const validationBytes = await readFile('/home/raffael/orca/workspaces/rimeflow-nn-validation/rfb-android-val-candidate-01/evidence/reports/android-litert-device-validation-report.json');
const manifestBytes = await readFile('/home/raffael/orca/workspaces/rimeflow-nn-base/rfb-android-device-accept-01-2/.device-evidence/rfb-android-device-accept-02/evidence-manifest.json');
const summaryBytes = await readFile('/home/raffael/orca/workspaces/rimeflow-nn-base/rfb-android-device-accept-01-2/.device-evidence/rfb-android-device-accept-02/execution-summary.json');
const validationReport = JSON.parse(validationBytes);
const deviceSummary = JSON.parse(summaryBytes);
const conformanceReport = JSON.parse(await readFile(resolve(root, 'reports/os6-base-litert-v2-conformance.json'), 'utf8'));
const runnerInventory = JSON.parse(await readFile(resolve(root, 'evidence/platform/runner-inventory.json'), 'utf8'));
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const cases = [];
const reject = (name, mutate, expected) => {
  const candidate = structuredClone(closure);
  mutate(candidate);
  try {
    validateJsonSchema(schema, candidate);
  } catch (error) {
    if (expected.test(String(error.message))) { cases.push(name); return; }
    throw error;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};

reject('supported overclaim', (candidate) => { candidate.supported = true; }, /const mismatch/);
reject('final close overclaim', (candidate) => { candidate.finalPlatformClose = true; }, /const mismatch/);
reject('performance promotion', (candidate) => { candidate.performancePassed = true; }, /const mismatch/);
reject('comparison outcome drift', (candidate) => { candidate.gates.performance.outcome = 'passed'; }, /const mismatch/);

validateRunnerInventory(runnerInventory);
const rejectRunnerOverclaim = (name, mutate) => {
  const candidate = structuredClone(runnerInventory);
  mutate(candidate.runners.find((runner) => runner.target === 'android-arm64-device'));
  try {
    validateRunnerInventory(candidate);
  } catch (error) {
    if (/blocked Android runner evidence boundary/.test(String(error.message))) { cases.push(name); return; }
    throw error;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};
rejectRunnerOverclaim('runner performance promotion', (runner) => { runner.observedEvidence.performancePassed = true; });
rejectRunnerOverclaim('runner support promotion', (runner) => { runner.observedEvidence.supported = true; });
rejectRunnerOverclaim('runner final close promotion', (runner) => { runner.observedEvidence.finalPlatformClose = true; });

const acceptedInputs = {
  closure,
  schema,
  validationReport,
  deviceSummary,
  conformanceReport,
  validationReportSha256: digest(validationBytes),
  deviceManifestSha256: digest(manifestBytes),
  executionSummarySha256: digest(summaryBytes),
};
validateAndroidBaseClosure(acceptedInputs);
const rejectEvidenceDrift = (name, mutate, expected) => {
  const candidate = structuredClone(closure);
  mutate(candidate);
  try {
    validateAndroidBaseClosure({ ...acceptedInputs, closure: candidate });
  } catch (error) {
    if (expected.test(String(error.message))) { cases.push(name); return; }
    throw error;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};
rejectEvidenceDrift('input binding/runtime name collapse', (candidate) => {
  candidate.ioContract.input.runtimeName = candidate.ioContract.input.signatureBindingName;
}, /input binding\/runtime tensor identity drift/);
rejectEvidenceDrift('output runtime name drift', (candidate) => {
  candidate.ioContract.output.runtimeName = 'output_0';
}, /output binding\/runtime tensor identity drift/);
console.log(JSON.stringify({ ok: true, negativeCases: cases }));
