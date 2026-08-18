#!/usr/bin/env node

import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const NDK = {
  revision: "27.2.12479018",
  url: "https://dl.google.com/android/repository/android-ndk-r27c-linux.zip",
  sha256: "59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc",
};
const LITERT_SDK = {
  version: "2.1.6",
  url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_cc_sdk.zip",
  sha256: "2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8",
};
const RUST_BINDING = {
  crate: "google-ai-edge-litert",
  version: "0.1.3",
  upstreamCrateSha256: "fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8",
};

export const ANDROID_TARGETS = Object.freeze({
  "android-arm64-v8a-api26-cpu": Object.freeze({
    arch: "arm64",
    cargoTarget: "aarch64-linux-android",
    compilerTarget: "aarch64-linux-android26",
    cmakeAbi: "arm64-v8a",
    elfMachine: 183,
    runtime: Object.freeze({
      url: "https://storage.googleapis.com/litert/binaries/2.1.6/android_arm64/libLiteRt.so",
      sha256: "35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369",
    }),
  }),
  "android-x86_64-api26-cpu": Object.freeze({
    arch: "x86_64",
    cargoTarget: "x86_64-linux-android",
    compilerTarget: "x86_64-linux-android26",
    cmakeAbi: "x86_64",
    elfMachine: 62,
    runtime: Object.freeze({
      url: "https://storage.googleapis.com/litert/binaries/2.1.6/android_x86_64/libLiteRt.so",
      sha256: "aa1530ba8b37b537d37139760716d183d2d7dc1f7781791ddf1d071c73eca535",
    }),
  }),
});

async function main() {
const args = parseArgs(process.argv.slice(2));

if (args.verify) {
  const manifestPath = path.join(args.verify, "bundle-manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  await verifyBundle(args.verify, manifest);
  console.log(JSON.stringify({ bundle: args.verify, verified: true, bundleId: manifest.bundleId }));
  process.exit(0);
}

for (const name of ["runner", "runtime", "buildProvenance", "artifact", "conversionManifest", "conversionManifestSha256", "fixtures", "out"]) {
  if (!args[name]) throw new Error(`缺少 --${toKebab(name)}`);
}

const conversionBytes = await readFile(args.conversionManifest);
const conversionSha256 = sha256(conversionBytes);
if (conversionSha256 !== args.conversionManifestSha256) {
  throw new Error(`conversion manifest SHA-256 不符：${conversionSha256}`);
}
const conversion = JSON.parse(conversionBytes.toString("utf8"));
if (conversion.schemaVersion !== 1 || conversion.runtime?.version !== "2.1.6") {
  throw new Error("仅接受 schema v1 / LiteRT 2.1.6 conversion manifest");
}

const artifact = await identity(args.artifact);
if (artifact.sha256 !== conversion.artifact?.sha256 || artifact.bytes !== conversion.artifact?.bytes) {
  throw new Error("外部 TFLite 与 conversion manifest 的 digest/长度不一致");
}
const runner = await identity(args.runner);
const runtime = await identity(args.runtime);
const provenanceData = JSON.parse(await readFile(args.buildProvenance, "utf8"));
const target = validateBuildProvenance(provenanceData, runtime.sha256);
await assertElfArchitecture(args.runner, target.elfMachine, "runner");
await assertElfArchitecture(args.runtime, target.elfMachine, "runtime");
const buildProvenance = await identity(args.buildProvenance);

const fixtureConfig = JSON.parse(await readFile(args.fixtures, "utf8"));
if (fixtureConfig.schemaVersion !== 1 || !Array.isArray(fixtureConfig.fixtures) || fixtureConfig.fixtures.length === 0) {
  throw new Error("fixture config 必须是非空 schema v1");
}
const fixtureInputs = [];
const fixtureIds = new Set();
for (const fixture of fixtureConfig.fixtures) {
  if (!/^[a-z0-9][a-z0-9-]*$/.test(fixture.id) || fixtureIds.has(fixture.id)) {
    throw new Error(`fixture ID 必须安全且唯一：${fixture.id}`);
  }
  fixtureIds.add(fixture.id);
  if (!["golden", "fault-never-promote"].includes(fixture.kind)) {
    throw new Error(`fixture ${fixture.id} kind 无效`);
  }
  const source = await identity(fixture.sourcePath);
  if (source.sha256 !== fixture.sha256) throw new Error(`fixture ${fixture.id} SHA-256 不符`);
  if (fixture.dtype !== "f32" || JSON.stringify(fixture.shape) !== "[1,3,640,640]" || source.bytes !== 4_915_200) {
    throw new Error(`fixture ${fixture.id} 不是冻结的 NCHW FP32 [1,3,640,640]`);
  }
  fixtureInputs.push({ ...fixture, source });
}

const contentIdentity = sha256(Buffer.from(JSON.stringify({
  runner: runner.sha256,
  runtime: runtime.sha256,
  conversionManifest: conversionSha256,
  artifact: artifact.sha256,
  buildProvenance: buildProvenance.sha256,
  fixtures: fixtureInputs.map((item) => [item.id, item.kind, item.source.sha256]),
  target: provenanceData.target,
})));
const bundleId = `sha256-${contentIdentity}`;
const root = path.join(args.out, bundleId);

for (const directory of ["bin", "lib", "manifest", "model", "inputs", "provenance", "outputs", "reports"]) {
  await mkdir(path.join(root, directory), { recursive: true });
}
const destinations = {
  runner: "bin/rimeflow-android-litert-runner",
  runtime: "lib/libLiteRt.so",
  buildProvenance: "provenance/android-runner-build.json",
  conversion: "provenance/litert-artifact-manifest.json",
  modelManifest: "manifest/model-manifest.json",
  artifact: "model/yolov8n-fp32.tflite",
};
await copyFile(args.runner, path.join(root, destinations.runner));
await copyFile(args.runtime, path.join(root, destinations.runtime));
await copyFile(args.buildProvenance, path.join(root, destinations.buildProvenance));
await copyFile(args.conversionManifest, path.join(root, destinations.conversion));
await copyFile(args.artifact, path.join(root, destinations.artifact));

const input = conversion.ioContract?.input;
const output = conversion.ioContract?.output;
if (!input || !output) throw new Error("conversion manifest 缺少 ioContract");
const modelManifest = {
  schemaVersion: 1,
  model: { id: "rimeflow-yolov8n", version: `litert-${artifact.sha256.slice(0, 16)}` },
  tensors: {
    inputs: [{ role: "image", name: input.name, index: 0, shape: input.shape, layout: "NCHW", dtype: "f32" }],
    outputs: [{ role: conversion.ioContract.outputRole, name: output.name, index: 0, shape: output.shape, layout: "NCHW", dtype: "f32" }],
  },
  artifacts: [{
    id: "yolov8n-litert-fp32",
    format: "tflite",
    targets: [{ os: "android", arch: target.arch }],
    path: destinations.artifact,
    sha256: artifact.sha256,
    converter: { name: "litert-torch", version: conversion.toolchain["litert-torch"] },
    inputs: ["image"],
    outputs: [conversion.ioContract.outputRole],
  }],
};
await writeFile(path.join(root, destinations.modelManifest), `${JSON.stringify(modelManifest, null, 2)}\n`);

const fixtures = [];
for (const fixture of fixtureInputs) {
  const relative = `inputs/${fixture.id}.f32le.bin`;
  await copyFile(fixture.sourcePath, path.join(root, relative));
  fixtures.push({
    id: fixture.id,
    kind: fixture.kind,
    input: await bundleIdentity(root, relative),
    role: fixture.role ?? "image",
    shape: fixture.shape,
    dtype: fixture.dtype,
  });
}

const manifest = {
  schemaVersion: 1,
  bundleId,
  target: { os: "android", arch: target.arch },
  minimumApi: 26,
  cpuOnly: true,
  runner: await bundleIdentity(root, destinations.runner),
  runtimeLibraries: [await bundleIdentity(root, destinations.runtime)],
  provenance: [
    await bundleIdentity(root, destinations.buildProvenance),
    await bundleIdentity(root, destinations.conversion),
  ],
  modelManifest: await bundleIdentity(root, destinations.modelManifest),
  artifact: await bundleIdentity(root, destinations.artifact),
  fixtures,
  gates: {
    initializationDeadlineMs: 30_000,
    goldenRuns: 2,
    performanceWarmupRuns: 5,
    performanceSampleRuns: 30,
    collectPeakRss: true,
    collectPackageLoad: true,
  },
};
const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(path.join(root, "bundle-manifest.json"), manifestBytes);
await writeFile(path.join(root, "bundle-manifest.sha256"), `${sha256(manifestBytes)}  bundle-manifest.json\n`);
await verifyBundle(root, manifest);
console.log(JSON.stringify({
  bundle: root,
  bundleId,
  manifestSha256: sha256(manifestBytes),
  artifactSha256: artifact.sha256,
  fixtureCount: fixtures.length,
}));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}

async function verifyBundle(root, manifest) {
  if (
    manifest.schemaVersion !== 1 ||
    manifest.target?.os !== "android" ||
    !["arm64", "x86_64"].includes(manifest.target?.arch)
  ) {
    throw new Error("bundle target/schema 无效");
  }
  if (manifest.minimumApi < 26 || manifest.cpuOnly !== true || manifest.gates?.goldenRuns !== 2) {
    throw new Error("bundle gate/CPU/API 合约无效");
  }
  if (manifest.runtimeLibraries?.length !== 1) {
    throw new Error("bundle 必须且只能包含一个目标架构的 libLiteRt.so");
  }
  const files = [
    manifest.runner,
    ...manifest.runtimeLibraries,
    ...manifest.provenance,
    manifest.modelManifest,
    manifest.artifact,
    ...manifest.fixtures.map((fixture) => fixture.input),
  ];
  for (const file of files) {
    if (path.isAbsolute(file.path) || file.path.split("/").includes("..")) throw new Error(`越界路径 ${file.path}`);
    const actual = await bundleIdentity(root, file.path);
    if (actual.sha256 !== file.sha256 || actual.bytes !== file.bytes) throw new Error(`bundle 文件不匹配 ${file.path}`);
  }

  const manifestBytes = await readFile(path.join(root, "bundle-manifest.json"));
  const checksum = (await readFile(path.join(root, "bundle-manifest.sha256"), "utf8")).trim();
  if (checksum !== `${sha256(manifestBytes)}  bundle-manifest.json`) {
    throw new Error("bundle manifest sidecar SHA-256 不匹配");
  }

  const buildFile = manifest.provenance.find((file) => file.path === "provenance/android-runner-build.json");
  const conversionFile = manifest.provenance.find((file) => file.path === "provenance/litert-artifact-manifest.json");
  if (!buildFile || !conversionFile) throw new Error("bundle 缺少 build/conversion provenance");
  const build = JSON.parse(await readFile(path.join(root, buildFile.path), "utf8"));
  const target = validateBuildProvenance(build, manifest.runtimeLibraries[0].sha256);
  if (manifest.target.arch !== target.arch) {
    throw new Error("bundle target 与 build provenance 架构不一致");
  }
  await assertElfArchitecture(path.join(root, manifest.runner.path), target.elfMachine, "runner");
  await assertElfArchitecture(path.join(root, manifest.runtimeLibraries[0].path), target.elfMachine, "runtime");

  const model = JSON.parse(await readFile(path.join(root, manifest.modelManifest.path), "utf8"));
  const modelArtifact = model.artifacts?.find((artifact) => artifact.sha256 === manifest.artifact.sha256);
  if (
    model.schemaVersion !== 1 ||
    !modelArtifact ||
    modelArtifact.path !== manifest.artifact.path ||
    modelArtifact.format !== "tflite" ||
    !modelArtifact.targets?.some((candidate) => candidate.os === "android" && candidate.arch === target.arch)
  ) {
    throw new Error(`model manifest 未以相同 identity 引用 Android ${target.arch} TFLite`);
  }

  const expectedBundleId = `sha256-${sha256(Buffer.from(JSON.stringify({
    runner: manifest.runner.sha256,
    runtime: manifest.runtimeLibraries[0].sha256,
    conversionManifest: conversionFile.sha256,
    artifact: manifest.artifact.sha256,
    buildProvenance: buildFile.sha256,
    fixtures: manifest.fixtures.map((fixture) => [fixture.id, fixture.kind, fixture.input.sha256]),
    target: build.target,
  })))}`;
  if (manifest.bundleId !== expectedBundleId || path.basename(path.resolve(root)) !== expectedBundleId) {
    throw new Error("bundle 目录名/ID 与内容 identity 不一致");
  }
}

export function validateBuildProvenance(provenance, runtimeSha256) {
  const target = ANDROID_TARGETS[provenance?.target];
  const explicitMapping = ["arch", "cargoTarget", "compilerTarget", "cmakeAbi"];
  const presentMappings = explicitMapping.filter((field) => provenance?.[field] !== undefined);
  const legacyArm64Mapping = target?.arch === "arm64" && presentMappings.length === 0;
  if (
    !target ||
    provenance.schemaVersion !== 1 ||
    (!legacyArm64Mapping && (
      presentMappings.length !== explicitMapping.length ||
      provenance.arch !== target.arch ||
      provenance.cargoTarget !== target.cargoTarget ||
      provenance.compilerTarget !== target.compilerTarget ||
      provenance.cmakeAbi !== target.cmakeAbi
    )) ||
    !sameLockedInput(provenance.ndk, NDK) ||
    !sameLockedInput(provenance.litertSdk, LITERT_SDK) ||
    provenance.litertRuntime?.version !== "2.1.6" ||
    provenance.litertRuntime?.url !== target.runtime.url ||
    provenance.litertRuntime?.sha256 !== target.runtime.sha256 ||
    runtimeSha256 !== target.runtime.sha256 ||
    !sameLockedInput(provenance.rustBinding, RUST_BINDING)
  ) {
    throw new Error("build provenance、runtime digest 与冻结 Android LiteRT 目标不一致");
  }
  return target;
}

function sameLockedInput(actual, expected) {
  return Object.entries(expected).every(([key, value]) => actual?.[key] === value);
}

export async function assertElfArchitecture(filePath, expectedMachine, label) {
  const bytes = await readFile(filePath);
  if (
    bytes.length < 20 ||
    bytes[0] !== 0x7f ||
    bytes.subarray(1, 4).toString("ascii") !== "ELF" ||
    bytes[4] !== 2 ||
    bytes[5] !== 1 ||
    bytes.readUInt16LE(18) !== expectedMachine
  ) {
    throw new Error(`${label} ELF 架构与 build provenance 不一致`);
  }
}

async function identity(filePath) {
  const bytes = await readFile(filePath);
  const metadata = await stat(filePath);
  if (!metadata.isFile()) throw new Error(`不是普通文件: ${filePath}`);
  return { sha256: sha256(bytes), bytes: bytes.length };
}

async function bundleIdentity(root, relative) {
  return { path: relative, ...(await identity(path.join(root, relative))) };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    if (!flag?.startsWith("--") || argv[index + 1] === undefined) throw new Error(`无效参数 ${flag ?? "<empty>"}`);
    result[toCamel(flag.slice(2))] = argv[index + 1];
  }
  return result;
}

function toCamel(value) {
  return value.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function toKebab(value) {
  return value.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}
