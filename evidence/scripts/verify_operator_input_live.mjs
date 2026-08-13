import { chmod, mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { validatePublication } from './evidence_contracts.mjs';
import { createControlledEnvironment, IGNORED_CALLER_BUILD_VARIABLES, OFFICIAL_LINUX_X64_TOOLS, runTrusted, verifyFreshRunner, verifyOfficialToolchain } from './official_live_trust.mjs';
import { createOperatorExport, restoreOwnerWrite } from './operator_input_export.mjs';
import { validateEvidence } from './validate_evidence.mjs';

const root = resolve(import.meta.dirname, '../..');
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
validatePublication(publication);
const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-operator-live-'));
const bundle = join(temporary, 'operator-export');
const cargoHome = join(temporary, 'cargo-home');
const target = join(temporary, 'target');
const scratch = join(temporary, 'validator-scratch');
const controlledEnv = createControlledEnvironment({ temporary, cargoHome, target });

try {
  console.error('live-stage: verify frozen Linux x64 tool trust root');
  const tools = await verifyOfficialToolchain();
  await Promise.all([mkdir(cargoHome), mkdir(target), mkdir(scratch)]);
  console.error('live-stage: fresh fetch and complete exact blob export');
  const exported = await createOperatorExport(bundle, publication, { git: OFFICIAL_LINUX_X64_TOOLS.git.path, gitEnv: controlledEnv, readOnly: true });
  console.error('live-stage: verify fresh target and build locked production runner');
  const runner = resolve(target, 'release/rimeflow-raw-golden');
  await verifyFreshRunner(runner, target, { mustNotExist: true });
  runTrusted(OFFICIAL_LINUX_X64_TOOLS.cargo.path, ['build', '--jobs', '1', '--locked', '--release', '--manifest-path', resolve(exported.source, 'evidence/tooling/raw-golden/Cargo.toml')], { cwd: exported.source, env: controlledEnv, inherit: true });
  await chmod(runner, 0o555);
  const runnerIdentity = await verifyFreshRunner(runner, target);
  console.error('live-stage: execute fresh runner and main evidence validation');
  const validation = await validateEvidence({ root, operatorRoot: exported.source, operatorBundleRoot: exported.bundle, operatorGit: OFFICIAL_LINUX_X64_TOOLS.git.path, operatorGitEnv: controlledEnv, productionRunner: runner, productionRunnerEnv: controlledEnv, scratchRoot: scratch });
  console.log(JSON.stringify({
    ok: true, repository: publication.operatorInputPublication.repository, ref: publication.operatorInputPublication.ref,
    fetchHead: exported.fetchHead, tree: exported.tree, exportedFileCount: exported.exportedFileCount, verified: exported.verified,
    fetchProcessGroup: exported.receipt.fetchProcessGroup,
    trustedTools: tools,
    controlledBuild: { callerBuildVariablesIgnored: IGNORED_CALLER_BUILD_VARIABLES, path: controlledEnv.PATH, rustc: controlledEnv.RUSTC, linker: controlledEnv.CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER, cargoHomeFresh: true, targetFresh: true, locked: true },
    sourcePermissions: exported.sourcePermissions, runner: runnerIdentity, ignoredCallerOperatorRoot: true, validation,
  }));
} catch (error) {
  console.error(`live verification failed/unavailable: ${error.message}`);
  process.exitCode = 1;
} finally {
  await restoreOwnerWrite(temporary);
  await rm(temporary, { recursive: true, force: true });
}
