import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

import { assertTrackedUnchanged, validatePublication } from './evidence_contracts.mjs';
import { OFFICIAL_LINUX_X64_TOOLS } from './official_live_trust.mjs';
import { verifyOperatorExport } from './operator_input_export.mjs';

const root = resolve(import.meta.dirname, '../..');
const manifest = JSON.parse(await readFile(resolve(root, 'evidence/replay/task1-replay.json'), 'utf8'));
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
validatePublication(publication);
const operatorBundle = process.env.RIMEFLOW_OPERATOR_BUNDLE;
if (!operatorBundle || !operatorBundle.startsWith('/')) throw new Error('ordinary replay requires absolute RIMEFLOW_OPERATOR_BUNDLE prepared by prepare_operator_input.mjs');
const operatorExport = await verifyOperatorExport(operatorBundle, publication, { git: OFFICIAL_LINUX_X64_TOOLS.git.path, gitEnv: { PATH: '/usr/bin:/bin', GIT_EXEC_PATH: '/usr/lib/git-core', LC_ALL: 'C' } });
const childEnv = { ...process.env, RIMEFLOW_OPERATOR_BUNDLE: operatorExport.bundle, RIMEFLOW_OPERATOR_ROOT: operatorExport.source };
const tracked = spawnSync(OFFICIAL_LINUX_X64_TOOLS.git.path, ['ls-files', '-z'], { cwd: root, encoding: 'buffer' });
if (tracked.status !== 0) throw new Error(tracked.stderr.toString());
const trackedPaths = tracked.stdout.toString().split('\0').filter(Boolean);
const snapshot = async () => Object.fromEntries(await Promise.all(trackedPaths.map(async (path) => {
  const bytes = await readFile(resolve(root, path));
  return [path, createHash('sha256').update(bytes).digest('hex')];
})));
const before = await snapshot();
const beforeDiff = spawnSync(OFFICIAL_LINUX_X64_TOOLS.git.path, ['diff', '--binary', 'HEAD', '--', ...trackedPaths], { cwd: root, encoding: 'buffer', maxBuffer: 32 * 1024 * 1024 });
if (beforeDiff.status !== 0) throw new Error(beforeDiff.stderr.toString());

for (const output of manifest.outputs) {
  const bytes = await readFile(resolve(root, output.path));
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (bytes.length !== output.bytes || digest !== output.sha256) throw new Error(`tracked evidence drift: ${output.path}`);
}

const commands = [
  ['main-validator', ['evidence/scripts/validate_evidence.mjs']],
  ['performance-negative', ['evidence/scripts/test_performance_validator_negative.mjs']],
  ['publication-schema-negative', ['evidence/scripts/test_publication_validator_negative.mjs']],
  ['replay-negative', ['evidence/scripts/test_replay_validator_negative.mjs']],
  ['main-security-negative', ['evidence/scripts/test_main_validator_security_negative.mjs']],
  ['official-trust-negative', ['evidence/scripts/test_official_live_trust_negative.mjs']],
];
const executed = [];
for (const [id, args] of commands) {
  const result = spawnSync(process.execPath, args, { cwd: root, env: childEnv, encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${id} failed:\n${result.stdout}\n${result.stderr}`);
  executed.push({ id, exitCode: result.status, stdoutSha256: createHash('sha256').update(result.stdout).digest('hex') });
}
if (JSON.stringify(executed.map((item) => item.id)) !== JSON.stringify(manifest.ordinaryReplayValidationChain)) throw new Error('ordinary replay validation chain drift');
const after = await snapshot();
const afterDiff = spawnSync(OFFICIAL_LINUX_X64_TOOLS.git.path, ['diff', '--binary', 'HEAD', '--', ...trackedPaths], { cwd: root, encoding: 'buffer', maxBuffer: 32 * 1024 * 1024 });
if (afterDiff.status !== 0) throw new Error(afterDiff.stderr.toString());
assertTrackedUnchanged(before, after, beforeDiff.stdout, afterDiff.stdout);
console.log(JSON.stringify({ ok: true, mode: 'ordinary-strict-chain', rounds: commands.length, executed, operatorExport: { bundle: operatorExport.bundle, commit: operatorExport.fetchHead, tree: operatorExport.tree, exportedFileCount: operatorExport.exportedFileCount, verifiedObjectCount: operatorExport.verified.length }, trackedEvidenceUnchanged: true, trackedFileCount: trackedPaths.length, outputCount: manifest.outputs.length }));
