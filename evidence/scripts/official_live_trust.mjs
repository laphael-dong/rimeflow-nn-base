import { createHash } from 'node:crypto';
import { lstat, readFile, readdir, readlink, realpath, stat } from 'node:fs/promises';
import { arch, platform } from 'node:os';
import { dirname } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

export const OFFICIAL_LINUX_X64_TOOLS = Object.freeze({
  node: Object.freeze({ path: '/home/raffael/.nvm/versions/node/v24.15.0/bin/node', sha256: 'd1de76d8edf2fededf6f8b30d244e2c0529ac607923a018283b77e9c74bd932c', versionArgs: ['--version'], version: 'v24.15.0', owner: 'current-user' }),
  git: Object.freeze({ path: '/usr/bin/git', sha256: '2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668', versionArgs: ['--version'], version: 'git version 2.43.0', owner: 'root' }),
  gitRemoteHttp: Object.freeze({ path: '/usr/lib/git-core/git-remote-http', sha256: '8353a5124ddb2281838bc54bd97c21cbecfe900615fdb60b281198c5058ddb5b', versionArgs: null, version: 'Git HTTPS transport resolved target', owner: 'root' }),
  cargo: Object.freeze({ path: '/home/raffael/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo', sha256: '828980723df339d62434390e9fb8ef8831036583343ae2316b7ab5646b5c1953', versionArgs: ['--version'], version: 'cargo 1.97.1 (c980f4866 2026-06-30)', owner: 'current-user' }),
  rustc: Object.freeze({ path: '/home/raffael/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc', sha256: 'd3a664c970a9fd8361b64194861bebc1ae37b9054e5ee3400dc1c9e691797eea', versionArgs: ['--version'], version: 'rustc 1.97.1 (8bab26f4f 2026-07-14)', owner: 'current-user' }),
  linker: Object.freeze({ path: '/usr/bin/x86_64-linux-gnu-gcc-13', sha256: '1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26', versionArgs: ['--version'], version: 'x86_64-linux-gnu-gcc-13 (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0', owner: 'root' }),
  collect2: Object.freeze({ path: '/usr/libexec/gcc/x86_64-linux-gnu/13/collect2', sha256: '4d1f341ae5b763b513258ee2812422a45e063c30a2f1924a0cf63d3699f3a158', versionArgs: ['--version'], version: 'collect2 version 13.3.0', versionStream: 'stderr', owner: 'root' }),
  ld: Object.freeze({ path: '/usr/bin/x86_64-linux-gnu-ld.bfd', sha256: 'e9ceb054c12207970f2726dfc07e9a66b411602748628baf27399f02a9bbb31b', versionArgs: ['--version'], version: 'GNU ld (GNU Binutils for Ubuntu) 2.42', owner: 'root' }),
  assembler: Object.freeze({ path: '/usr/bin/x86_64-linux-gnu-as', sha256: '21aff249b692b5c31a44007491f922dcb49f41323e362c57d2ada3f52eddb7f0', versionArgs: ['--version'], version: 'GNU assembler (GNU Binutils for Ubuntu) 2.42', owner: 'root' }),
  ar: Object.freeze({ path: '/usr/bin/x86_64-linux-gnu-ar', sha256: '6452af2eea333b8c65e1adb92964fc8f97863ab003fa13f9d12bff5345cd7dbe', versionArgs: ['--version'], version: 'GNU ar (GNU Binutils for Ubuntu) 2.42', owner: 'root' }),
  ranlib: Object.freeze({ path: '/usr/bin/x86_64-linux-gnu-ranlib', sha256: '60254978b8ee2c1b21b41d16a18a621dbbcc72e43eb2fa9f916818256c77e9ee', versionArgs: ['--version'], version: 'GNU ranlib (GNU Binutils for Ubuntu) 2.42', owner: 'root' }),
});

export const GIT_HTTPS_LAUNCHER = Object.freeze({
  path: '/usr/lib/git-core/git-remote-https',
  linkTarget: 'git-remote-http',
  canonicalPath: OFFICIAL_LINUX_X64_TOOLS.gitRemoteHttp.path,
});

export const IGNORED_CALLER_BUILD_VARIABLES = Object.freeze([
  'PATH', 'RUSTC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'RUSTFLAGS',
  'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTC', 'CARGO_TARGET_DIR',
  'CARGO_HOME', 'RUSTUP_HOME', 'RUSTUP_TOOLCHAIN', 'GIT_EXEC_PATH',
  'GIT_CONFIG_GLOBAL', 'GIT_CONFIG_SYSTEM', 'GIT_CONFIG_COUNT',
]);

const firstLine = (value) => value.trim().split(/\r?\n/, 1)[0];

export function runTrusted(program, args, { cwd, env, encoding = 'utf8', inherit = false, timeoutMs } = {}) {
  const result = spawnSync(program, args, {
    cwd, env, encoding,
    stdio: inherit ? ['ignore', 'inherit', 'inherit'] : undefined,
    maxBuffer: 64 * 1024 * 1024, timeout: timeoutMs,
  });
  if (result.error || result.status !== 0) {
    throw new Error(`${program} ${args.join(' ')} failed: ${result.error?.message ?? ''}\n${result.stdout ?? ''}\n${result.stderr ?? ''}`);
  }
  return result.stdout ?? '';
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const bounded = async (promise, timeoutMs, label) => {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${label} did not complete within ${timeoutMs}ms`)), timeoutMs); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

const waitForExitOrTimeout = async (exitPromise, timeoutMs) => {
  let timer;
  const timeoutToken = Symbol('timeout');
  try {
    return await Promise.race([
      exitPromise,
      new Promise((resolve) => { timer = setTimeout(() => resolve(timeoutToken), timeoutMs); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
};

export async function listProcessGroupMembers(pgid) {
  if (platform() !== 'linux' || !Number.isSafeInteger(pgid) || pgid <= 1) throw new Error('process-group inspection requires a task-specific Linux PGID');
  const members = [];
  for (const name of await readdir('/proc')) {
    if (!/^\d+$/.test(name)) continue;
    try {
      const record = await readFile(`/proc/${name}/stat`, 'utf8');
      const end = record.lastIndexOf(')');
      if (end < 0) continue;
      const fields = record.slice(end + 2).split(' ');
      if (Number(fields[2]) === pgid) members.push({ pid: Number(name), ppid: Number(fields[1]), pgid, state: fields[0] });
    } catch (error) {
      if (!['ENOENT', 'ESRCH'].includes(error.code)) throw error;
    }
  }
  return members.sort((left, right) => left.pid - right.pid);
}

const waitForEmptyProcessGroup = async (pgid, timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  let members = await listProcessGroupMembers(pgid);
  while (members.length > 0 && Date.now() < deadline) {
    await delay(20);
    members = await listProcessGroupMembers(pgid);
  }
  return members;
};

const signalProcessGroup = (pgid, signal) => {
  try {
    process.kill(-pgid, signal);
    return { attempted: true, delivered: true, signal, error: null };
  } catch (error) {
    if (error.code === 'ESRCH') return { attempted: true, delivered: false, signal, error: 'ESRCH' };
    return { attempted: true, delivered: false, signal, error: `${error.code ?? error.name}: ${error.message}` };
  }
};

const collectProcessOutput = (stream, chunks, mirror, limit, state) => {
  stream.on('data', (chunk) => {
    state.bytes += chunk.length;
    if (state.bytes <= limit) chunks.push(chunk);
    else state.overflow = true;
    if (mirror) mirror.write(chunk);
  });
};

export async function runTrustedProcessGroup(program, args, {
  cwd, env, timeoutMs, termGraceMs = 1000, killGraceMs = 1000,
  closeGraceMs = 1000, inherit = false, maxBuffer = 64 * 1024 * 1024,
} = {}) {
  if (platform() !== 'linux') throw new Error('trusted process-group execution is supported only on Linux');
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('trusted process-group execution requires a positive timeout');
  const child = spawn(program, args, { cwd, env, detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
  const leaderPid = child.pid;
  const pgid = Number.isSafeInteger(leaderPid) && leaderPid > 1 ? leaderPid : null;
  const stdout = [];
  const stderr = [];
  const outputState = { bytes: 0, overflow: false };
  collectProcessOutput(child.stdout, stdout, inherit ? process.stdout : null, maxBuffer, outputState);
  collectProcessOutput(child.stderr, stderr, inherit ? process.stderr : null, maxBuffer, outputState);

  let spawnError = null;
  let exitCode = null;
  let exitSignal = null;
  let closed = false;
  const exitPromise = new Promise((resolve) => {
    child.once('error', (error) => { spawnError = error; resolve({ kind: 'error' }); });
    child.once('exit', (code, signal) => { exitCode = code; exitSignal = signal; resolve({ kind: 'exit' }); });
  });
  const closePromise = new Promise((resolve) => child.once('close', () => { closed = true; resolve(); }));
  if (pgid === null) {
    await bounded(exitPromise, closeGraceMs, 'failed spawn error');
    await bounded(closePromise, closeGraceMs, 'failed spawn close');
    const receipt = { program, args, leaderPid: null, pgid: null, timedOut: false, exitCode, exitSignal, spawnError: spawnError ? `${spawnError.code ?? spawnError.name}: ${spawnError.message}` : 'spawn returned no safe leader PID', outputOverflow: outputState.overflow, sigterm: { attempted: false, delivered: false, signal: 'SIGTERM', error: null }, sigkill: { attempted: false, delivered: false, signal: 'SIGKILL', error: null }, closed, membersBeforeCleanup: [], membersAfterTerm: [], membersAfterKill: [], remainingMembers: [], cleanupComplete: closed };
    const error = new Error(`${program} ${args.join(' ')} failed before process-group creation: processGroup=${JSON.stringify(receipt)}`);
    error.processGroup = receipt;
    throw error;
  }
  const first = await waitForExitOrTimeout(exitPromise, timeoutMs);
  const timedOut = typeof first === 'symbol';

  let membersBeforeCleanup = await listProcessGroupMembers(pgid);
  const needsCleanup = timedOut || spawnError !== null || exitCode !== 0 || exitSignal !== null || outputState.overflow || membersBeforeCleanup.length > 0;
  let sigterm = { attempted: false, delivered: false, signal: 'SIGTERM', error: null };
  let sigkill = { attempted: false, delivered: false, signal: 'SIGKILL', error: null };
  let membersAfterTerm = membersBeforeCleanup;
  let membersAfterKill = membersBeforeCleanup;
  const cleanupErrors = [];

  if (needsCleanup && membersBeforeCleanup.length > 0) {
    sigterm = signalProcessGroup(pgid, 'SIGTERM');
    if (sigterm.error && sigterm.error !== 'ESRCH') cleanupErrors.push(`SIGTERM: ${sigterm.error}`);
    membersAfterTerm = await waitForEmptyProcessGroup(pgid, termGraceMs);
    membersAfterKill = membersAfterTerm;
    if (membersAfterTerm.length > 0) {
      sigkill = signalProcessGroup(pgid, 'SIGKILL');
      if (sigkill.error && sigkill.error !== 'ESRCH') cleanupErrors.push(`SIGKILL: ${sigkill.error}`);
      membersAfterKill = await waitForEmptyProcessGroup(pgid, killGraceMs);
    }
  }

  try { await bounded(exitPromise, closeGraceMs, 'direct child exit'); } catch (error) { cleanupErrors.push(error.message); }
  try { await bounded(closePromise, closeGraceMs, 'direct child close'); } catch (error) { cleanupErrors.push(error.message); }
  const remainingMembers = await listProcessGroupMembers(pgid);
  if (remainingMembers.length > 0) cleanupErrors.push(`process group still has members: ${JSON.stringify(remainingMembers)}`);
  if (!closed) cleanupErrors.push('direct child close was not observed');

  const receipt = {
    program, args, leaderPid, pgid, timedOut, exitCode, exitSignal,
    spawnError: spawnError ? `${spawnError.code ?? spawnError.name}: ${spawnError.message}` : null,
    outputOverflow: outputState.overflow, sigterm, sigkill, closed,
    membersBeforeCleanup, membersAfterTerm, membersAfterKill, remainingMembers,
    cleanupComplete: cleanupErrors.length === 0 && remainingMembers.length === 0 && closed,
  };
  const stdoutBuffer = Buffer.concat(stdout);
  const stderrBuffer = Buffer.concat(stderr);
  if (timedOut || spawnError || exitCode !== 0 || exitSignal || outputState.overflow || membersBeforeCleanup.length > 0 || !receipt.cleanupComplete) {
    const reason = timedOut ? `timed out after ${timeoutMs}ms` : spawnError ? `spawn failed: ${receipt.spawnError}` : outputState.overflow ? `output exceeded ${maxBuffer} bytes` : membersBeforeCleanup.length > 0 && exitCode === 0 ? 'successful leader left process-group descendants' : `exited code=${exitCode} signal=${exitSignal}`;
    const error = new Error(`${program} ${args.join(' ')} failed: ${reason}; processGroup=${JSON.stringify(receipt)}; cleanupErrors=${JSON.stringify(cleanupErrors)}\n${stdoutBuffer.toString()}\n${stderrBuffer.toString()}`);
    error.processGroup = receipt;
    throw error;
  }
  return { stdout: stdoutBuffer, stderr: stderrBuffer, processGroup: receipt };
}

const verifyParentDirectory = async (path, expectedUid, expectedGid = null) => {
  const parentPath = dirname(path);
  const parent = await stat(parentPath);
  const insecureWriteMask = expectedUid === 0 ? 0o022 : 0o002;
  if (!parent.isDirectory() || parent.uid !== expectedUid || (expectedGid !== null && parent.gid !== expectedGid) || (parent.mode & insecureWriteMask) !== 0) throw new Error(`official tool parent directory ownership/mode mismatch: ${parentPath}`);
  return { path: parentPath, uid: parent.uid, gid: parent.gid, mode: (parent.mode & 0o777).toString(8) };
};

export async function verifyExecutableIdentity(name, expected, { uid = process.getuid() } = {}) {
  const link = await lstat(expected.path);
  if (!link.isFile() || link.isSymbolicLink()) throw new Error(`official tool ${name} must be a non-symlink regular file`);
  const canonical = await realpath(expected.path);
  if (canonical !== expected.path) throw new Error(`official tool ${name} canonical path mismatch`);
  const metadata = await stat(canonical);
  const expectedUid = expected.owner === 'root' ? 0 : uid;
  if (metadata.uid !== expectedUid || (metadata.mode & 0o022) !== 0 || (metadata.mode & 0o111) === 0) throw new Error(`official tool ${name} ownership/mode mismatch`);
  const bytes = await readFile(canonical);
  const digest = sha256(bytes);
  if (digest !== expected.sha256) throw new Error(`official tool ${name} SHA-256 mismatch`);
  let version = expected.version;
  if (expected.versionArgs) {
    const result = spawnSync(canonical, expected.versionArgs, { env: { PATH: '/usr/bin:/bin', LC_ALL: 'C' }, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 });
    if (result.error || result.status !== 0) throw new Error(`official tool ${name} version command failed`);
    version = firstLine(expected.versionStream === 'stderr' ? result.stderr : result.stdout || result.stderr);
    if (version !== expected.version) throw new Error(`official tool ${name} version mismatch`);
  }
  return { name, path: expected.path, canonicalPath: canonical, version, bytes: bytes.length, sha256: digest, uid: metadata.uid, mode: (metadata.mode & 0o777).toString(8), parentDirectory: await verifyParentDirectory(canonical, expectedUid, expectedUid === 0 ? null : process.getgid()) };
}

export async function verifyGitHttpsLauncher(expected = GIT_HTTPS_LAUNCHER) {
  const link = await lstat(expected.path);
  if (!link.isSymbolicLink() || link.uid !== 0) throw new Error('official Git HTTPS launcher must be a root-owned symlink');
  const linkTarget = await readlink(expected.path);
  const canonical = await realpath(expected.path);
  if (linkTarget !== expected.linkTarget || canonical !== expected.canonicalPath) throw new Error('official Git HTTPS launcher target mismatch');
  return {
    name: 'gitRemoteHttpsLauncher', path: expected.path, fileType: 'symbolic-link', linkTarget,
    canonicalPath: canonical, uid: link.uid, parentDirectory: await verifyParentDirectory(expected.path, 0),
  };
}

export async function verifyOfficialToolchain(expectedTools = OFFICIAL_LINUX_X64_TOOLS) {
  if (platform() !== 'linux' || arch() !== 'x64') throw new Error('official live toolchain supports only frozen Linux x64');
  if (process.execPath !== expectedTools.node.path) throw new Error('official Node executable path mismatch');
  const receipt = {};
  for (const [name, expected] of Object.entries(expectedTools)) receipt[name] = await verifyExecutableIdentity(name, expected);
  receipt.gitRemoteHttpsLauncher = await verifyGitHttpsLauncher();
  return receipt;
}

export function createControlledEnvironment({ temporary, cargoHome, target, callerEnv = process.env }) {
  const env = {
    PATH: '/usr/libexec/gcc/x86_64-linux-gnu/13:/usr/bin:/bin', HOME: temporary, LC_ALL: 'C',
    CARGO_HOME: cargoHome, CARGO_TARGET_DIR: target, CARGO_BUILD_JOBS: '1', CARGO_INCREMENTAL: '0',
    CARGO_NET_GIT_FETCH_WITH_CLI: 'false', CARGO_PROFILE_RELEASE_DEBUG: '0', RUST_BACKTRACE: '0',
    RUSTC: OFFICIAL_LINUX_X64_TOOLS.rustc.path,
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER: OFFICIAL_LINUX_X64_TOOLS.linker.path,
    CC: OFFICIAL_LINUX_X64_TOOLS.linker.path, AR: OFFICIAL_LINUX_X64_TOOLS.ar.path,
    RANLIB: OFFICIAL_LINUX_X64_TOOLS.ranlib.path, GIT_EXEC_PATH: dirname(GIT_HTTPS_LAUNCHER.path),
    GIT_TERMINAL_PROMPT: '0', GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null',
  };
  for (const name of ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy', 'SSL_CERT_FILE', 'SSL_CERT_DIR']) {
    if (callerEnv[name]) env[name] = callerEnv[name];
  }
  return env;
}

export async function verifyFreshRunner(runner, target, { mustNotExist = false } = {}) {
  let metadata;
  try { metadata = await lstat(runner); } catch (error) {
    if (mustNotExist && error.code === 'ENOENT') return { absent: true };
    throw error;
  }
  if (mustNotExist) throw new Error('fresh target runner existed before build');
  const canonicalTarget = await realpath(target);
  const canonicalRunner = await realpath(runner);
  if (!canonicalRunner.startsWith(`${canonicalTarget}/`) || canonicalRunner !== runner) throw new Error('runner canonical path escaped fresh target or used a symlink');
  if (!metadata.isFile() || metadata.isSymbolicLink() || (metadata.mode & 0o111) === 0) throw new Error('runner must be a regular executable');
  if (metadata.uid !== process.getuid() || (metadata.mode & 0o222) !== 0) throw new Error('runner ownership/mode mismatch');
  const bytes = await readFile(runner);
  return { path: runner, canonicalPath: canonicalRunner, fileType: 'regular-executable', hostOs: platform(), hostArchitecture: arch(), bytes: bytes.length, sha256: sha256(bytes), uid: metadata.uid, mode: (metadata.mode & 0o777).toString(8) };
}
