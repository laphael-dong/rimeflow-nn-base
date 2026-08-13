import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { assertTrackedUnchanged, validateReplayManifest } from './evidence_contracts.mjs';

const root = resolve(import.meta.dirname, '../..');
const load = async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'));
const replay = await load('evidence/replay/task1-replay.json');
const publication = await load('evidence/publication/task1-publication.json');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const cases = [];
const reject = (name, mutate, pattern) => {
  const candidate = structuredClone(replay);
  mutate(candidate);
  try { validateReplayManifest(candidate, publication, sha256); } catch (error) {
    if (!pattern.test(error.message)) throw error;
    cases.push(name);
    return;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
};

reject('deleted step', (x) => x.steps.pop(), /step set/);
reject('forged failed round', (x) => x.steps[0].rounds[0].stderr = 'forged failure', /step digest/);
reject('modified exit code', (x) => x.steps[0].rounds[0].exitCode = 1, /step digest/);
reject('modified input digest', (x) => x.steps[0].rounds[0].inputs[0].sha256 = '0'.repeat(64), /step digest/);
reject('modified output digest', (x) => x.steps[0].rounds[0].outputs[0].sha256 = '0'.repeat(64), /step digest/);
reject('modified log digest', (x) => x.steps[0].rounds[0].log.sha256 = '0'.repeat(64), /step digest/);
reject('deterministic false', (x) => x.steps[0].repeatComparison.deterministicOutputDigestsEqual = false, /step digest/);
reject('tracked mutation claim', (x) => x.ordinaryReplayMutatesTrackedEvidence = true, /mutation claim/);
reject('performance determinism false', (x) => x.twoRoundPerformanceOutputDigestsEqual = false, /performance determinism/);
reject('repository identity drift', (x) => x.steps[0].rounds[0].repositoryHead = '0'.repeat(40), /step digest/);
reject('empty output ledger', (x) => { x.outputs = []; }, /output ledger length/);
reject('missing Schema output', (x) => { x.outputs.shift(); }, /output ledger length/);
reject('extra output entry', (x) => { x.outputs.push(structuredClone(x.outputs.at(-1))); }, /output ledger length/);
reject('duplicate output path', (x) => { x.outputs[1].path = x.outputs[0].path; }, /duplicate output path/);
reject('output path traversal', (x) => { x.outputs[0].path = '../platform-matrix.schema.json'; }, /path traversal/);
reject('wrong output bytes', (x) => { x.outputs[0].bytes += 1; }, /output tuples/);
reject('wrong output SHA', (x) => { x.outputs[0].sha256 = '0'.repeat(64); }, /output tuples/);
reject('reordered outputs', (x) => { [x.outputs[0], x.outputs[1]] = [x.outputs[1], x.outputs[0]]; }, /output tuples/);
reject('runner owner drift', (x) => { x.runner.owner = 'fabricated'; }, /runner identity/);
reject('runner OS drift', (x) => { x.runner.os = 'fabricated'; }, /runner identity/);
reject('formal CI overclaim', (x) => { x.immutableLogEvidence.formalCiState = 'passed'; }, /log\/CI identity/);
reject('unknown top-level field', (x) => { x.requiredCiState = 'passed'; }, /top-level fields/);
reject('fabricated repository HEAD', (x) => { x.repositoryHeadAtReplay = '0'.repeat(40); }, /top-level fields/);
try {
  assertTrackedUnchanged({ 'tracked.json': 'before' }, { 'tracked.json': 'after' }, Buffer.from(''), Buffer.from('modified'));
  throw new Error('tracked mutation negative case unexpectedly passed');
} catch (error) {
  if (!/modified tracked evidence/.test(error.message)) throw error;
  cases.push('actual tracked content mutation');
}

console.log(JSON.stringify({ ok: true, negativeCaseCount: cases.length, negativeCases: cases }));
