import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

import { createControlledEnvironment, GIT_HTTPS_LAUNCHER, IGNORED_CALLER_BUILD_VARIABLES, OFFICIAL_LINUX_X64_TOOLS, runTrusted, verifyExecutableIdentity, verifyFreshRunner, verifyGitHttpsLauncher, verifyOfficialToolchain } from './official_live_trust.mjs';

const scratch = await mkdtemp(resolve(tmpdir(), 'rimeflow-task16-fix02-trust-negative-'));
const cases = [];
const reject = async (name, action, pattern) => {
  try { await action(); } catch (error) {
    if (!pattern.test(error.message)) throw error;
    cases.push(name);
    return;
  }
  throw new Error(`security case unexpectedly passed: ${name}`);
};

try {
  const hostileBin = resolve(scratch, 'hostile-bin');
  const marker = resolve(scratch, 'hostile-marker');
  await mkdir(hostileBin);
  for (const name of ['git', 'tar', 'chmod', 'cargo', 'rustc', 'rustc-wrapper', 'rustc-workspace-wrapper']) await writeFile(resolve(hostileBin, name), `#!/bin/sh\nprintf '%s\\n' '${name}' >> '${marker}'\nexit 0\n`, { mode: 0o700 });
  const callerEnv = { PATH: `${hostileBin}:/usr/bin:/bin`, RUSTC: resolve(hostileBin, 'rustc'), RUSTC_WRAPPER: resolve(hostileBin, 'rustc-wrapper'), RUSTC_WORKSPACE_WRAPPER: resolve(hostileBin, 'rustc-workspace-wrapper'), RUSTFLAGS: '-C linker=/hostile/linker', CARGO_ENCODED_RUSTFLAGS: '-C\x1flinker=/hostile/linker', CARGO_BUILD_RUSTC: resolve(hostileBin, 'rustc'), CARGO_TARGET_DIR: resolve(scratch, 'hostile-target'), CARGO_HOME: resolve(scratch, 'hostile-cargo-home'), RUSTUP_HOME: resolve(scratch, 'hostile-rustup-home'), RUSTUP_TOOLCHAIN: 'hostile-toolchain', GIT_EXEC_PATH: hostileBin, GIT_CONFIG_GLOBAL: resolve(scratch, 'hostile-gitconfig'), GIT_CONFIG_SYSTEM: resolve(scratch, 'hostile-system-gitconfig'), GIT_CONFIG_COUNT: '1' };
  const env = createControlledEnvironment({ temporary: scratch, cargoHome: resolve(scratch, 'cargo-home'), target: resolve(scratch, 'target'), callerEnv });
  for (const name of IGNORED_CALLER_BUILD_VARIABLES) if (name !== 'PATH' && env[name] === callerEnv[name]) throw new Error(`caller build injection survived: ${name}`);
  if (env.PATH.includes(hostileBin) || env.RUSTC !== OFFICIAL_LINUX_X64_TOOLS.rustc.path) throw new Error('caller PATH/RUSTC survived controlled environment');
  await verifyOfficialToolchain();
  runTrusted(OFFICIAL_LINUX_X64_TOOLS.git.path, ['--version'], { env });
  runTrusted(OFFICIAL_LINUX_X64_TOOLS.cargo.path, ['--version'], { env });
  runTrusted(OFFICIAL_LINUX_X64_TOOLS.rustc.path, ['--version'], { env });
  try { await readFile(marker); throw new Error('hostile PATH marker was executed'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  cases.push('hostile caller PATH and Rust/Cargo injection ignored without marker execution');

  const fakeTool = resolve(scratch, 'fake-git');
  await writeFile(fakeTool, `#!/bin/sh\nprintf fake >> '${marker}'\n`, { mode: 0o700 });
  const fakeLink = resolve(scratch, 'git-link');
  await symlink(fakeTool, fakeLink);
  await reject('symlink-substituted tool rejected before execution', () => verifyExecutableIdentity('git', { ...OFFICIAL_LINUX_X64_TOOLS.git, path: fakeLink }), /non-symlink regular file/);
  try { await readFile(marker); throw new Error('symlink tool marker was executed'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await reject('tool SHA mismatch', () => verifyExecutableIdentity('git', { ...OFFICIAL_LINUX_X64_TOOLS.git, sha256: '0'.repeat(64) }), /SHA-256 mismatch/);
  await reject('tool version mismatch', () => verifyExecutableIdentity('git', { ...OFFICIAL_LINUX_X64_TOOLS.git, version: 'git version hostile' }), /version mismatch/);
  await verifyGitHttpsLauncher();
  cases.push('legitimate root-owned Git HTTPS launcher accepted with exact target identity');
  await reject('Git HTTPS launcher target mismatch', () => verifyGitHttpsLauncher({ ...GIT_HTTPS_LAUNCHER, linkTarget: 'hostile' }), /target mismatch/);

  const target = resolve(scratch, 'runner-target');
  await mkdir(resolve(target, 'release'), { recursive: true });
  const runner = resolve(target, 'release/rimeflow-raw-golden');
  await writeFile(runner, '#!/bin/sh\n', { mode: 0o500 });
  await reject('runner pre-creation rejected', () => verifyFreshRunner(runner, target, { mustNotExist: true }), /existed before build/);
  await rm(runner);
  const outside = resolve(scratch, 'outside-runner');
  await writeFile(outside, '#!/bin/sh\n', { mode: 0o500 });
  await symlink(outside, runner);
  await reject('runner symlink/substitution rejected', () => verifyFreshRunner(runner, target), /escaped fresh target|symlink/);

  console.log(JSON.stringify({ ok: true, negativeCaseCount: cases.length, negativeCases: cases, ignoredCallerBuildVariables: IGNORED_CALLER_BUILD_VARIABLES }));
} finally {
  await rm(scratch, { recursive: true, force: true });
}
