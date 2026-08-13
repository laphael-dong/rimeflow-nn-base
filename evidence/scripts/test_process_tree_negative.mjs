import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

import { listProcessGroupMembers, runTrustedProcessGroup } from './official_live_trust.mjs';

const scratch = await mkdtemp(resolve(tmpdir(), 'rimeflow-task16-fix03-process-tree-'));
const fixture = resolve(import.meta.dirname, 'process_tree_fixture.mjs');
const cases = [];
const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
const markerSize = async (path) => {
  try { return (await readFile(path)).length; } catch (error) { if (error.code === 'ENOENT') return 0; throw error; }
};
const assertMarkerStopped = async (path) => {
  const before = await markerSize(path);
  await delay(80);
  const after = await markerSize(path);
  if (before === 0 || after !== before) throw new Error(`marker continued after process-group cleanup: ${path}`);
};
const reject = async (name, scenario, expected, options = {}) => {
  const marker = resolve(scratch, `${String(cases.length + 1).padStart(2, '0')}-${scenario}.marker`);
  let receipt;
  try {
    await runTrustedProcessGroup(process.execPath, [fixture, scenario, marker], { timeoutMs: 120, termGraceMs: 100, killGraceMs: 500, closeGraceMs: 500, ...options });
  } catch (error) {
    if (!expected.test(error.message) || !error.processGroup) throw error;
    receipt = error.processGroup;
  }
  if (!receipt) throw new Error(`process-tree case unexpectedly passed: ${name}`);
  if (!receipt.cleanupComplete || receipt.remainingMembers.length !== 0 || (await listProcessGroupMembers(receipt.pgid)).length !== 0) throw new Error(`process-tree case left PGID members: ${name}`);
  await assertMarkerStopped(marker);
  cases.push({ name, leaderPid: receipt.leaderPid, pgid: receipt.pgid, timedOut: receipt.timedOut, exitCode: receipt.exitCode, exitSignal: receipt.exitSignal, sigterm: receipt.sigterm, sigkill: receipt.sigkill, cleanupComplete: receipt.cleanupComplete, remainingMemberCount: receipt.remainingMembers.length });
};

let sentinel;
try {
  await reject('parent child and grandchild terminate after timeout', 'tree-term', /timed out/);
  await reject('SIGTERM-ignoring process tree is killed after grace period', 'tree-ignore-term', /timed out/);
  if (!cases.at(-1).sigkill.attempted || !cases.at(-1).sigkill.delivered) throw new Error('SIGKILL fallback was not exercised');
  await reject('early-failing leader still reaps continuing descendants', 'early-fail', /exited code=7/);
  await reject('zero-exit helper cannot leave a continuing descendant', 'zero-with-descendant', /successful leader left process-group descendants/);
  await reject('signal-terminated leader still reaps continuing descendants', 'signal-with-descendant', /exited code=null signal=SIGHUP/);
  await reject('output-overflow exception still reaps continuing descendants', 'overflow-with-descendant', /output exceeded 64 bytes/, { maxBuffer: 64 });
  const missingProgram = resolve(scratch, 'missing-program');
  try {
    await runTrustedProcessGroup(missingProgram, [], { timeoutMs: 100, termGraceMs: 100, killGraceMs: 100, closeGraceMs: 500 });
    throw new Error('spawn-failure case unexpectedly passed');
  } catch (error) {
    if (!/failed before process-group creation/.test(error.message) || !error.processGroup?.cleanupComplete || error.processGroup.leaderPid !== null || error.processGroup.pgid !== null || error.processGroup.remainingMembers.length !== 0) throw error;
    cases.push({ name: 'spawn failure closes without creating a process group', leaderPid: null, pgid: null, spawnError: error.processGroup.spawnError, cleanupComplete: true, remainingMemberCount: 0 });
  }

  const sentinelMarker = resolve(scratch, 'sentinel.marker');
  sentinel = spawn(process.execPath, [fixture, 'sentinel', sentinelMarker], { detached: true, stdio: 'ignore' });
  sentinel.unref();
  await delay(60);
  const successMarker = resolve(scratch, 'success.marker');
  const successStartedAt = Date.now();
  const success = await runTrustedProcessGroup(process.execPath, [fixture, 'success', successMarker], { timeoutMs: 10000, termGraceMs: 100, killGraceMs: 100, closeGraceMs: 500 });
  const successElapsedMs = Date.now() - successStartedAt;
  process.kill(sentinel.pid, 0);
  const sentinelBefore = await markerSize(sentinelMarker);
  await delay(80);
  const sentinelAfter = await markerSize(sentinelMarker);
  if (sentinelAfter <= sentinelBefore || success.processGroup.pgid === sentinel.pid) throw new Error('normal success disturbed the out-of-group sentinel');
  if (!success.processGroup.cleanupComplete || success.processGroup.remainingMembers.length !== 0 || successElapsedMs >= 1000) throw new Error('normal success process group did not close promptly and cleanly');
  cases.push({ name: 'normal success clears timeout and does not signal out-of-group sentinel', leaderPid: success.processGroup.leaderPid, pgid: success.processGroup.pgid, sentinelPid: sentinel.pid, successElapsedMs, cleanupComplete: true, remainingMemberCount: 0 });

  console.log(JSON.stringify({ ok: true, negativeCaseCount: cases.length, cases, scratchPrefix: 'rimeflow-task16-fix03-process-tree-', allMarkersStoppedAfterCleanup: true, allProcessGroupsEmpty: true }));
} finally {
  if (sentinel?.pid) {
    try { process.kill(-sentinel.pid, 'SIGTERM'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    for (let attempt = 0; attempt < 50; attempt += 1) {
      if ((await listProcessGroupMembers(sentinel.pid)).length === 0) break;
      await delay(20);
    }
    if ((await listProcessGroupMembers(sentinel.pid)).length !== 0) {
      try { process.kill(-sentinel.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
      await delay(40);
    }
  }
  await rm(scratch, { recursive: true, force: true });
}
