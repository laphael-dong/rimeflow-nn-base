import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { validatePublication } from './evidence_contracts.mjs';
import { createControlledEnvironment, OFFICIAL_LINUX_X64_TOOLS, verifyOfficialToolchain } from './official_live_trust.mjs';
import { createOperatorExport } from './operator_input_export.mjs';

const root = resolve(import.meta.dirname, '../..');
const target = process.argv[2];
if (!target || !target.startsWith('/')) throw new Error('usage: node evidence/scripts/prepare_operator_input.mjs /absolute/new/bundle/path');
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
validatePublication(publication);
await verifyOfficialToolchain();
const controlledEnv = createControlledEnvironment({ temporary: resolve(target, '..'), cargoHome: resolve(target, '../unused-cargo-home'), target: resolve(target, '../unused-target') });
const exported = await createOperatorExport(target, publication, { git: OFFICIAL_LINUX_X64_TOOLS.git.path, gitEnv: controlledEnv, readOnly: true });
console.log(JSON.stringify({ ok: true, bundle: exported.bundle, source: exported.source, repository: exported.receipt.repository, ref: exported.receipt.ref, commit: exported.fetchHead, tree: exported.tree, exportedFileCount: exported.exportedFileCount, verifiedObjectCount: exported.verified.length, sourcePermissions: exported.sourcePermissions, fetchProcessGroup: exported.receipt.fetchProcessGroup }));
