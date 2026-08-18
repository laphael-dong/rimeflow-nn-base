import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  ANDROID_TARGETS,
  assertElfArchitecture,
  validateBuildProvenance,
} from "./package_bundle.mjs";

const buildScript = path.resolve(import.meta.dirname, "build_bundle.sh");

function producerContract(arch) {
  return JSON.parse(execFileSync(buildScript, ["--arch", arch, "--print-contract"], { encoding: "utf8" }));
}

function provenance(contract) {
  return {
    schemaVersion: 1,
    target: contract.provenanceTarget,
    arch: contract.bundleArch,
    cargoTarget: contract.cargoTarget,
    compilerTarget: contract.compilerTarget,
    cmakeAbi: contract.cmakeAbi,
    ndk: {
      revision: "27.2.12479018",
      url: "https://dl.google.com/android/repository/android-ndk-r27c-linux.zip",
      sha256: "59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc",
    },
    litertSdk: {
      version: "2.1.6",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_cc_sdk.zip",
      sha256: "2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8",
    },
    litertRuntime: contract.litertRuntime,
    rustBinding: {
      crate: "google-ai-edge-litert",
      version: "0.1.3",
      upstreamCrateSha256: "fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8",
    },
  };
}

test("producer and packager agree on both frozen Android targets", () => {
  for (const arch of ["arm64", "x86_64"]) {
    const contract = producerContract(arch);
    const target = validateBuildProvenance(provenance(contract), contract.litertRuntime.sha256);
    assert.deepEqual(target, ANDROID_TARGETS[contract.provenanceTarget]);
    assert.equal(target.arch, contract.bundleArch);
  }
});

test("producer rejects an unapproved Android target", () => {
  const result = spawnSync(buildScript, ["--arch", "riscv64", "--print-contract"], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /仅允许 arm64 或 x86_64/);
});

test("packager rejects cross-architecture runtime and provenance", () => {
  const x86 = producerContract("x86_64");
  const data = provenance(x86);
  assert.throws(
    () => validateBuildProvenance(data, ANDROID_TARGETS["android-arm64-v8a-api26-cpu"].runtime.sha256),
    /runtime digest/,
  );
  data.compilerTarget = "aarch64-linux-android26";
  assert.throws(() => validateBuildProvenance(data, x86.litertRuntime.sha256), /冻结 Android LiteRT 目标/);
});

test("legacy arm64 schema v1 provenance remains valid", () => {
  const arm64 = producerContract("arm64");
  const data = provenance(arm64);
  for (const field of ["arch", "cargoTarget", "compilerTarget", "cmakeAbi"]) delete data[field];
  assert.equal(validateBuildProvenance(data, arm64.litertRuntime.sha256).arch, "arm64");

  data.arch = "arm64";
  assert.throws(() => validateBuildProvenance(data, arm64.litertRuntime.sha256), /冻结 Android LiteRT 目标/);
});

test("ELF machine must match the selected Android architecture", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "rimeflow-android-target-"));
  try {
    const elf = Buffer.alloc(64);
    elf.set(Buffer.from([0x7f, 0x45, 0x4c, 0x46, 2, 1]));
    elf.writeUInt16LE(62, 18);
    const file = path.join(root, "runner");
    await writeFile(file, elf);
    await assertElfArchitecture(file, 62, "runner");
    await assert.rejects(assertElfArchitecture(file, 183, "runner"), /ELF 架构/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
