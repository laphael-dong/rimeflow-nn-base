import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { validatePerformanceContract, validatePlatformMatrix, validatePublication, validateThresholds } from './evidence_contracts.mjs';
import { validateFrozenPlatformSchema, validateJsonSchema } from './json_schema_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const load = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const publication = await load('evidence/publication/task1-publication.json');
const thresholds = await load('evidence/performance/backend-thresholds.json');
const matrix = await load('evidence/platform/platform-matrix.json');
const schema = await load('evidence/schemas/platform-matrix.schema.json');
const schemaBytes = await readFile(resolve(root, 'evidence/schemas/platform-matrix.schema.json'));
const performance = await load('evidence/performance/linux-x86_64-baseline.json');
const conversion = publication.operatorInputPublication.objects.find((object) => object.id === 'conversion-summary').sha256;
const fixedNow = new Date('2026-08-13T00:00:00.000Z');
const cases = [];
const reject = (name, pattern, value, mutate, validate) => {
  const candidate = structuredClone(value);
  mutate(candidate);
  try { validate(candidate); } catch (error) {
    if (!pattern.test(error.message)) throw error;
    cases.push(name);
    return;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};
const rejectThreshold = (name, mutate, pattern = /threshold|policy|approval|product metric/) => reject(name, pattern, thresholds, mutate, (x) => validateThresholds(x, undefined, fixedNow));
const rejectObject = (name, mutate, pattern = /operator publication/) => reject(name, pattern, publication, mutate, validatePublication);
const rejectSchema = (name, mutate, pattern = /JSON schema|unsupported platform/) => reject(name, pattern, matrix, mutate, (x) => { validateJsonSchema(schema, x); validatePlatformMatrix(x, conversion); });
const rejectFrozenSchema = (name, candidate, pattern = /frozen platform Schema/) => {
  try { validateFrozenPlatformSchema(Buffer.from(`${JSON.stringify(candidate)}\n`), candidate); } catch (error) {
    if (!pattern.test(error.message)) throw error;
    cases.push(name);
    return;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};

validateFrozenPlatformSchema(schemaBytes, schema);
rejectFrozenSchema('empty tracked Schema', {});
const weakenedSchema = structuredClone(schema);
weakenedSchema.required = [];
rejectFrozenSchema('weakened tracked Schema', weakenedSchema);
const unsupportedKeywordSchema = structuredClone(schema);
unsupportedKeywordSchema.unevaluatedProperties = false;
rejectFrozenSchema('unsupported-keyword tracked Schema', unsupportedKeywordSchema, /unsupported Schema keyword/);

rejectObject('stale operator commit', (x) => { x.operatorInputPublication.commit = '0'.repeat(40); }, /operator commit/);
rejectObject('missing operator object', (x) => { x.operatorInputPublication.objects.pop(); });
rejectObject('duplicate operator id', (x) => { x.operatorInputPublication.objects[1].id = x.operatorInputPublication.objects[0].id; }, /duplicate id/);
rejectObject('duplicate operator path', (x) => { x.operatorInputPublication.objects[1].path = x.operatorInputPublication.objects[0].path; }, /duplicate path/);
rejectObject('unknown operator object', (x) => { x.operatorInputPublication.objects[0].id = 'unknown'; }, /exact tuple/);
rejectObject('valid wrong blob', (x) => { x.operatorInputPublication.objects[0].blob = '0'.repeat(40); }, /exact tuple/);
rejectObject('path traversal', (x) => { x.operatorInputPublication.objects[0].path = '../models/yolov8n.onnx'; }, /path traversal/);
rejectObject('tree object', (x) => { x.operatorInputPublication.objects[0].type = 'tree'; }, /mode\/type/);
rejectObject('symlink mode', (x) => { x.operatorInputPublication.objects[0].mode = '120000'; }, /mode\/type/);
rejectObject('operator object extra field', (x) => { x.operatorInputPublication.objects[0].verified = true; }, /exact fields/);
rejectObject('five duplicate conversion summaries', (x) => { x.operatorInputPublication.objects = Array.from({ length: 5 }, () => structuredClone(x.operatorInputPublication.objects[4])); }, /duplicate id/);
rejectObject('production source missing', (x) => { x.operatorInputPublication.productionSources.pop(); }, /production source object set length/);
rejectObject('base self reference', (x) => { x.basePublicationState.commit = '0'.repeat(40); }, /self-reference/);
rejectObject('false CI success', (x) => { x.requiredCiState = 'passed'; }, /formal CI/);

rejectThreshold('deleted initialization threshold', (x) => { delete x.latencyAndMemoryThresholds.nativeInitialization; }, /frozen latency/);
rejectThreshold('tampered threshold number', (x) => { x.latencyAndMemoryThresholds.warmInferenceP95.absoluteMaxMs = 1001; }, /frozen latency/);
for (const [name, value] of [['null', null], ['NaN', Number.NaN], ['Infinity', Number.POSITIVE_INFINITY], ['negative', -1], ['zero', 0]]) {
  rejectThreshold(`threshold ${name}`, (x) => { x.latencyAndMemoryThresholds.nativeInitialization.relativeToWebMax = value; }, /frozen latency/);
}
for (const policy of ['nonFiniteMetricPolicy', 'mixedDeviceComparisonPolicy', 'mixedExecutionProviderPolicy', 'mixedDigestPolicy']) rejectThreshold(`relaxed ${policy}`, (x) => { x.hardGates[policy] = 'ignore'; }, new RegExp(policy));
rejectThreshold('deleted product metric', (x) => { x.externalProductMetrics.metrics.pop(); }, /external product metric contract/);
rejectThreshold('wrong product owner', (x) => { x.externalProductMetrics.owner = 'base'; }, /external product metric contract/);
rejectThreshold('external product passed overclaim', (x) => { x.externalProductMetrics.passed = true; }, /external product metric contract/);
rejectThreshold('cleared approval requiredFields', (x) => { x.approvalRule.requiredFields = []; }, /approval rule/);
rejectThreshold('approval over 90 policy', (x) => { x.approvalRule.maximumLifetimeDays = 365; }, /approval rule/);
rejectThreshold('approval self allowed', (x) => { x.approvalRule.selfApprovalPolicy = 'allow'; }, /approval failure policies/);
rejectThreshold('approval missing allowed', (x) => { x.approvalRule.missingApprovalPolicy = 'allow'; }, /approval failure policies/);
rejectThreshold('approval expired allowed', (x) => { x.approvalRule.expiredApprovalPolicy = 'allow'; }, /approval failure policies/);
rejectThreshold('approval cross scope allowed', (x) => { x.approvalRule.crossScopeReusePolicy = 'allow'; }, /approval failure policies/);

rejectSchema('schema missing minimum OS', (x) => { delete x.platforms[0].minimumOsVersion; });
rejectSchema('schema missing timeout', (x) => { delete x.platforms[0].timeoutsMs; });
rejectSchema('schema missing adapter', (x) => { delete x.platforms[0].adapter; });
rejectSchema('schema missing CI job', (x) => { delete x.platforms[0].requiredCiJob; });
rejectSchema('schema missing governance', (x) => { delete x.governance.runnerAdministration; });
rejectSchema('schema unknown field', (x) => { x.platforms[0].unknown = true; });
rejectSchema('schema wrong timeout type', (x) => { x.platforms[0].timeoutsMs.nativeInitialization = '15000'; });
rejectSchema('schema invalid calendar date', (x) => { x.platforms[0].officialVersionEvidence.checkedOn = '2026-02-30'; }, /calendar date/);
rejectSchema('hidden supported claim', (x) => { x.platforms[0].state = 'supported'; }, /unsupported platform claim/);
reject('missing platform', /platform set/, matrix, (x) => { x.platforms.pop(); }, (x) => validatePlatformMatrix(x, conversion));
reject('duplicate platform', /platform set/, matrix, (x) => { x.platforms[1].id = x.platforms[0].id; }, (x) => validatePlatformMatrix(x, conversion));

reject('cross-backend provider equality', /cross-backend provider/, performance, (x) => { x.comparability.crossBackendProviderEqual = true; }, (x) => validatePerformanceContract(x, publication));
reject('mixed Web provider round', /Web provider round/, performance, (x) => { x.comparability.providerIdentity.web.roundProviders[1] = 'webgpu'; }, (x) => validatePerformanceContract(x, publication));
reject('mixed Native provider round', /Native provider round/, performance, (x) => { x.comparability.providerIdentity.native.roundProviders[1] = 'Cuda'; }, (x) => validatePerformanceContract(x, publication));
reject('mixed digest', /mixed digest/, performance, (x) => { x.comparability.sameModelDigest = false; }, (x) => validatePerformanceContract(x, publication));
reject('blocked interpreted as passed', /blocked product metric state/, performance, (x) => { x.packageSizeMetrics.finalPackage.state = 'passed'; }, (x) => validatePerformanceContract(x, publication));

const validApproval = {
  approver: 'reviewer', submitter: 'author', platform: 'linux-x86_64-ort-cpu', metric: 'warmInferenceMs',
  candidateCommit: '1'.repeat(40), modelSha256: '2'.repeat(64), artifactDigest: '3'.repeat(64),
  runtimeName: 'onnxruntime', runtimeVersion: '1.24.3', observed: 101, threshold: 100,
  reason: 'Accepted bounded regression', createdAt: '2026-08-01T00:00:00.000Z', expiresAt: '2026-09-01T00:00:00.000Z',
};
const scopeFields = ['platform', 'metric', 'candidateCommit', 'modelSha256', 'artifactDigest', 'runtimeName', 'runtimeVersion'];
const approvalScope = Object.fromEntries(scopeFields.map((field) => [field, validApproval[field]]));
validateThresholds(thresholds, [validApproval], fixedNow, approvalScope);
const approvalCases = [
  ['approval missing field', /approval missing platform/, (x) => { delete x.platform; }],
  ['approval self approval', /self-approval/, (x) => { x.approver = x.submitter; }],
  ['approval expired', /expired/, (x) => { x.createdAt = '2026-01-01T00:00:00.000Z'; x.expiresAt = '2026-02-01T00:00:00.000Z'; }],
  ['approval over 90 days', /lifetime/, (x) => { x.expiresAt = '2026-12-01T00:00:00.000Z'; }],
  ['approval cross model', /scope mismatch: modelSha256/, (x) => { x.modelSha256 = '4'.repeat(64); }],
  ['approval cross artifact', /scope mismatch: artifactDigest/, (x) => { x.artifactDigest = '5'.repeat(64); }],
  ['approval cross candidate', /scope mismatch: candidateCommit/, (x) => { x.candidateCommit = '6'.repeat(40); }],
  ['approval cross runtime name', /scope mismatch: runtimeName/, (x) => { x.runtimeName = 'other-runtime'; }],
  ['approval cross runtime version', /scope mismatch: runtimeVersion/, (x) => { x.runtimeVersion = '9.9.9'; }],
  ['approval cross platform', /scope mismatch: platform/, (x) => { x.platform = 'windows-x86_64-windowsml'; }],
  ['approval cross metric', /scope mismatch: metric/, (x) => { x.metric = 'coldInferenceMs'; }],
  ['approval extra scope field', /approval exact fields/, (x) => { x.packageDigest = '7'.repeat(64); }],
];
for (const [name, pattern, mutate] of approvalCases) reject(name, pattern, validApproval, mutate, (x) => validateThresholds(thresholds, [x], fixedNow, approvalScope));

console.log(JSON.stringify({ ok: true, validSyntheticApproval: true, negativeCaseCount: cases.length, negativeCases: cases }));
