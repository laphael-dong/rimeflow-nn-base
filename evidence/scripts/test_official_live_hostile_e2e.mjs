import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

import { IGNORED_CALLER_BUILD_VARIABLES, OFFICIAL_LINUX_X64_TOOLS } from './official_live_trust.mjs';

const root = resolve(import.meta.dirname, '../..');
const scratch = await mkdtemp(join(tmpdir(), 'rimeflow-task16-fix02-live-hostile-'));
const hostileBin = resolve(scratch, 'hostile-bin');
const marker = resolve(scratch, 'executed-marker');
try {
  await mkdir(hostileBin);
  for (const name of ['git', 'tar', 'chmod', 'cargo', 'rustc', 'rustc-wrapper', 'rustc-workspace-wrapper']) await writeFile(resolve(hostileBin, name), `#!/bin/sh\nprintf '%s\\n' '${name}' >> '${marker}'\nexit 0\n`, { mode: 0o700 });
  const env = { ...process.env, PATH: `${hostileBin}:/usr/bin:/bin`, RUSTC: resolve(hostileBin, 'rustc'), RUSTC_WRAPPER: resolve(hostileBin, 'rustc-wrapper'), RUSTC_WORKSPACE_WRAPPER: resolve(hostileBin, 'rustc-workspace-wrapper'), RUSTFLAGS: '-C linker=/hostile/linker', CARGO_ENCODED_RUSTFLAGS: '-C\x1flinker=/hostile/linker', CARGO_BUILD_RUSTC: resolve(hostileBin, 'rustc'), CARGO_TARGET_DIR: resolve(scratch, 'hostile-target'), CARGO_HOME: resolve(scratch, 'hostile-cargo-home'), RUSTUP_HOME: resolve(scratch, 'hostile-rustup-home'), RUSTUP_TOOLCHAIN: 'hostile-toolchain', GIT_EXEC_PATH: hostileBin, GIT_CONFIG_GLOBAL: resolve(scratch, 'hostile-gitconfig'), GIT_CONFIG_SYSTEM: resolve(scratch, 'hostile-system-gitconfig'), GIT_CONFIG_COUNT: '1', RIMEFLOW_OPERATOR_ROOT: resolve(scratch, 'hostile-operator-root') };
  const result = spawnSync(OFFICIAL_LINUX_X64_TOOLS.node.path, ['evidence/scripts/verify_operator_input_live.mjs'], { cwd: root, env, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(`hostile official-live failed:\n${result.stdout}\n${result.stderr}`);
  try { await readFile(marker); throw new Error('hostile caller executable marker was executed'); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const receipt = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1));
  if (receipt.ok !== true || receipt.fetchHead !== 'c90d3957fbbd04b3f0b29eff7bc873b70eed4400' || receipt.tree !== '341d8b00fb5d4d9afeac856418950c1faa408b2e') throw new Error('official-live receipt repository identity mismatch');
  if (receipt.validation?.validationMode !== 'official-live-fresh-runner' || receipt.ignoredCallerOperatorRoot !== true) throw new Error('official-live receipt mode mismatch');
  if (receipt.sourcePermissions?.allEntriesNonWritable !== true || receipt.runner?.fileType !== 'regular-executable' || receipt.runner.mode !== '555') throw new Error('official-live permission/runner receipt mismatch');
  if (JSON.stringify(receipt.controlledBuild?.callerBuildVariablesIgnored) !== JSON.stringify(IGNORED_CALLER_BUILD_VARIABLES)) throw new Error('official-live ignored variable receipt mismatch');
  console.log(JSON.stringify({ ok: true, hostileMarkerExecuted: false, fetchHead: receipt.fetchHead, tree: receipt.tree, exportedFileCount: receipt.exportedFileCount, validationMode: receipt.validation.validationMode, sourcePermissions: receipt.sourcePermissions, runner: receipt.runner, trustedTools: receipt.trustedTools, ignoredCallerBuildVariables: receipt.controlledBuild.callerBuildVariablesIgnored }));
} finally {
  await rm(scratch, { recursive: true, force: true });
}
