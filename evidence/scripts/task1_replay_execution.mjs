import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { hostname, userInfo } from 'node:os';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';

export const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function execute(root, executable, args, env = {}) {
  return spawnSync(executable, args, { cwd: root, env: { ...process.env, ...env }, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
}

function git(root, args) {
  const result = execute(root, 'git', args);
  if (result.status !== 0) throw new Error(result.stderr || `git ${args.join(' ')} failed`);
  return result.stdout.trim();
}

async function snapshot(root, paths) {
  return Promise.all(paths.map(async (path) => {
    try {
      const bytes = await readFile(resolve(root, path));
      const metadata = await stat(resolve(root, path));
      return { path, exists: true, bytes: metadata.size, sha256: sha256(bytes) };
    } catch (error) {
      if (error.code === 'ENOENT') return { path, exists: false, bytes: null, sha256: null };
      throw error;
    }
  }));
}

function worktree(root) {
  return { tracked: git(root, ['status', '--short', '--untracked-files=no']), full: git(root, ['status', '--short']) };
}

export function runnerIdentity() {
  return { id: hostname(), owner: userInfo().username, os: process.platform, architecture: process.arch };
}

export function toolVersion(root, executable, args) {
  const result = execute(root, executable, args);
  return { command: [executable, ...args], exitCode: result.status, value: result.status === 0 ? result.stdout.trim() : null, stderr: result.status === 0 ? '' : result.stderr.trim() };
}

export async function runRepeatedStep({ root, id, command, executable, args, env = {}, inputPaths = [], outputPaths = [], runs = 2 }) {
  const rounds = [];
  for (let index = 0; index < runs; index += 1) {
    const startedAt = new Date().toISOString();
    const repositoryHead = git(root, ['rev-parse', 'HEAD']);
    const worktreeBefore = worktree(root);
    const inputs = await snapshot(root, inputPaths);
    const result = execute(root, executable, args, env);
    const outputs = await snapshot(root, outputPaths);
    const worktreeAfter = worktree(root);
    const endedAt = new Date().toISOString();
    const stdout = result.stdout ?? '';
    const stderr = result.stderr ?? '';
    rounds.push({
      run: index + 1,
      actualCommand: command,
      startedAt,
      endedAt,
      exitCode: result.status,
      signal: result.signal,
      repositoryHead,
      runnerId: hostname(),
      worktreeBefore,
      worktreeAfter,
      inputs,
      outputs,
      log: { storage: 'embedded-in-replay-manifest', stdout, stderr, bytes: Buffer.byteLength(stdout) + Buffer.byteLength(stderr), sha256: sha256(Buffer.from(`${stdout}\0${stderr}`)) },
    });
  }
  const outputDigestSets = rounds.map((round) => round.outputs.map((item) => `${item.path}:${item.sha256}`).join('|'));
  return {
    id,
    command,
    executed: true,
    blockedReason: null,
    rounds,
    repeatComparison: { runs: rounds.length, allExitCodesZero: rounds.every((round) => round.exitCode === 0), deterministicOutputDigestsEqual: new Set(outputDigestSets).size === 1 },
  };
}

export function blockedStep(id, command, blockedReason) {
  return { id, command, executed: false, blockedReason, rounds: [], repeatComparison: null };
}
