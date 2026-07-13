// rimeflow-onnx-base — inline JS template for operator crates.
//
// Owned by `rimeflow-onnx-base`. Operator crates' `build.rs` calls
// `rimeflow_onnx_base::build_helper::generate_extern_block(&cfg)` which
// reads this file, substitutes two sentinels (see build_helper source for
// the exact names — they are not repeated here to keep the single-occurrence
// contract) with the operator's shaders/preprocess.wgsl (escaped) and its
// DST_SIZE integer literal,
// and emits a full `#[wasm_bindgen(inline_js = "...")]` extern block to
// ${OUT_DIR}/ort_bridge_generated.rs. The operator crate's own
// `src/ort_bridge.rs` picks it up via `include!`.
//
// Do not import anything from the surrounding wasm-bindgen module — this file
// must self-contain (wasm-bindgen inline_js runs as a separate ES module).

// ─── GPUDevice interception (monkey-patch requestDevice) ───
// Guarantees that wgpu (when it lazily requests a device) and ORT WebGPU EP
// (when we set `env.webgpu.device`) share the SAME GPUDevice. Required for
// zero-copy fromGpuBuffer.
let _capturedDevice = null;
let _origRequestDevice = null;
let _deviceShared = false;

export function capture_webgpu_device() {
    if (_origRequestDevice || typeof GPUAdapter === 'undefined') return;
    _origRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function(...args) {
        if (_capturedDevice) return _capturedDevice;
        const device = await _origRequestDevice.apply(this, args);
        _capturedDevice = device;
        return device;
    };
}

// ─── ORT Session lifecycle ───
let _session    = null;
let _inputName  = null;
let _outputName = null;
let _loadedModelUrl = null;

async function _ensureOrtLoaded() {
    if (typeof globalThis.ort !== 'undefined') return;
    await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = '/ort/ort.min.js';
        s.onload  = resolve;
        s.onerror = () => reject(new Error('Failed to load /ort/ort.min.js'));
        document.head.appendChild(s);
    });
    if (typeof globalThis.ort === 'undefined') {
        throw new Error('onnxruntime-web loaded but globalThis.ort is undefined');
    }
}

export async function ort_init(model_url) {
    // Idempotent: consumers may call ort_init on every playback start / method
    // switch. Skip re-download + re-JIT when the same model is already loaded.
    // If a caller genuinely wants to swap models, `ort_release()` first.
    if (_session && _loadedModelUrl === model_url) {
        return;
    }
    if (_session) {
        try { _session.release(); } catch (_) { /* ignore */ }
        _session = null;
        _inputName = null;
        _outputName = null;
    }
    await _ensureOrtLoaded();
    globalThis.ort.env.wasm.wasmPaths = '/ort/';
    globalThis.ort.env.wasm.numThreads = 1;
    // r6 revisit (2026-07-12): try the OFFICIAL device-injection API.
    //
    // ORT WebGPU docs say: assign `ort.env.webgpu.device = <GPUDevice>`
    // BEFORE creating ANY WebGPU InferenceSession. Once any WebGPU session
    // exists, the property becomes read-only. This is different from the
    // per-session `executionProviders: [{name:'webgpu', device: ...}]` we
    // tried in r5 — that field is silently ignored.
    //
    // Also try `env.webgpu.adapter` for good measure (some ORT versions
    // use that as a hint if it's set before requestDevice).
    let deviceInjected = _deviceShared;
    // Skip re-assign if a prior ort_init already succeeded — ORT locks
    // `env.webgpu.device` after the first WebGPU session creation, so
    // any later write throws a TypeError. `_deviceShared === true` is
    // proof enough that our device is already installed.
    if (_capturedDevice && !_deviceShared) {
        try {
            globalThis.ort.env.webgpu.device = _capturedDevice;
            deviceInjected = true;
        } catch (e) {
            console.warn('[ort-bridge] env.webgpu.device assign failed:', e && e.message);
        }
    }
    _deviceShared = deviceInjected;

    const resp = await fetch(model_url);
    if (!resp.ok) throw new Error('Model fetch failed: ' + resp.status + ' ' + model_url);
    const buf = await resp.arrayBuffer();
    // No per-session device field (documented not to be honored). Rely on
    // env.webgpu.device being set above + monkey-patch as safety net.
    _session = await globalThis.ort.InferenceSession.create(buf, {
        executionProviders: ['webgpu', 'wasm'],
        preferredOutputLocation: 'cpu',
    });
    _inputName  = _session.inputNames[0];
    _outputName = _session.outputNames[0];
    _loadedModelUrl = model_url;
    // ── Device identity diagnostic ──
    // ORT lazily creates its device on `InferenceSession.create(...)`. Read
    // env.webgpu.device now and compare it to the wgpu-side device we
    // captured. If they are the SAME OBJECT, Tier A (`fromGpuBuffer` on
    // wgpu-allocated buffers) is possible. If different, ORT ignored our
    // shim and built its own — the r5 assumption.
    const ortDeviceMaybe = globalThis.ort.env.webgpu.device;
    const ortDevice = (ortDeviceMaybe && typeof ortDeviceMaybe.then === 'function')
        ? await ortDeviceMaybe
        : ortDeviceMaybe;
    const sameObject = !!(_capturedDevice && ortDevice && _capturedDevice === ortDevice);
    // Expose on globalThis so users can re-check in DevTools:
    //   __rimeflow_debug_device
    globalThis.__rimeflow_debug_device = {
        capturedDevice: _capturedDevice,
        ortEnvDevice:   ortDevice,
        sameObject,
        capturedLabel:  _capturedDevice && _capturedDevice.label,
        ortLabel:       ortDevice && ortDevice.label,
    };
    console.log(
        `[ort-bridge][diag] device shared=${sameObject} captured.label="${_capturedDevice && _capturedDevice.label}" ort.label="${ortDevice && ortDevice.label}"`,
    );

    // ── Adapter identity check ──
    // On Windows Chrome ignores powerPreference (crbug.com/369219127) and
    // may hand our compositor an iGPU adapter on multi-GPU laptops. That
    // silently caps ORT WebGPU EP throughput at 3-5x below the dGPU. Ask
    // the captured device's adapter what it actually is.
    try {
        if (_capturedDevice?.adapterInfo) {
            const ai = _capturedDevice.adapterInfo;
            console.log(
                `[ort-bridge][diag] adapter vendor="${ai.vendor}" architecture="${ai.architecture}" device="${ai.device}" description="${ai.description}"`,
            );
        } else if (globalThis.navigator?.gpu?.requestAdapter) {
            const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            const ai = a && (a.info || (typeof a.requestAdapterInfo === 'function' ? await a.requestAdapterInfo() : null));
            if (ai) {
                console.log(
                    `[ort-bridge][diag] freshly-requested high-perf adapter vendor="${ai.vendor}" architecture="${ai.architecture}" device="${ai.device}" description="${ai.description}"`,
                );
            }
        }
    } catch (e) {
        console.log('[ort-bridge][diag] adapter info probe failed:', e && e.message);
    }
    console.log(
        '[ort-bridge] session ready, input=' + _inputName +
        ', deviceCaptured=' + !!_capturedDevice +
        ', deviceSharedWithOrt=' + sameObject,
    );

    // ── fromGpuBuffer end-to-end probe ──
    // If _deviceShared is true, run a one-shot inference on a small
    // known-nonzero GPUBuffer to determine whether fromGpuBuffer + session.run
    // ACTUALLY consumes the data (nonzero output) or silently ignores it
    // (zero output, i.e. the r6 bug). This gives us a decisive answer
    // independent of preprocessing, canvas source, or 4K image sizes.
    if (_deviceShared && _capturedDevice && globalThis.ort.Tensor.fromGpuBuffer) {
        try {
            const dev = _capturedDevice;
            const sizeBytes = 1 * 3 * DST_SIZE * DST_SIZE * 4;
            const probeBuf = dev.createBuffer({
                size: sizeBytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            // Fill with 0.5 via writeBuffer — a value that (if consumed by
            // the model at all) would produce nonzero output tensor entries
            // for YOLOv8n (unnormalized 0.5s look like gray input, class
            // scores will be small but definitely nonzero).
            const filler = new Float32Array(sizeBytes / 4);
            filler.fill(0.5);
            dev.queue.writeBuffer(probeBuf, 0, filler);

            const probeTensor = globalThis.ort.Tensor.fromGpuBuffer(probeBuf, {
                dataType: 'float32',
                dims: [1, 3, DST_SIZE, DST_SIZE],
            });
            const probeResult = await _session.run({ [_inputName]: probeTensor });
            const probeOut = await probeResult[_outputName].getData();

            // Check: is the output all zeros? If yes → r6 bug (ORT ignored
            // our device's buffer). If nonzero → device sharing actually
            // works and we can enable useGpu in ort_detect.
            let nonzero = 0, maxAbs = 0;
            for (let i = 0; i < probeOut.length; i++) {
                if (probeOut[i] !== 0) nonzero++;
                const a = Math.abs(probeOut[i]);
                if (a > maxAbs) maxAbs = a;
            }
            const outFrac = nonzero / probeOut.length;
            globalThis.__rimeflow_debug_device.probe = {
                outLen:     probeOut.length,
                nonzeroFrac: outFrac,
                maxAbs,
                verdict:    outFrac > 0.01 ? 'PASS (fromGpuBuffer works)' : 'FAIL (output all-zero, r6 bug)',
            };
            console.log(
                '[ort-bridge][diag] fromGpuBuffer probe:',
                outFrac > 0.01 ? '✅ PASS' : '❌ FAIL',
                `out.len=${probeOut.length} nonzero=${(outFrac * 100).toFixed(1)}% maxAbs=${maxAbs.toExponential(2)}`,
            );
            probeBuf.destroy();
        } catch (e) {
            console.warn('[ort-bridge][diag] fromGpuBuffer probe threw:', e && e.message);
            globalThis.__rimeflow_debug_device.probe = { error: String(e && e.message) };
        }
    }
}

export function ort_release() {
    if (_session) { _session.release(); _session = null; _inputName = null; _outputName = null; }
    if (_cvOutBuf) { _cvOutBuf.destroy(); _cvOutBuf = null; }
    if (_cvUniBuf) { _cvUniBuf.destroy(); _cvUniBuf = null; }
    if (_cvInputTex) { _cvInputTex.destroy(); _cvInputTex = null; }
    _cvInputTexView = null;
    _cvSampler = null;
    _cvPipeline = null;
    _deviceShared = false;
    _loadedModelUrl = null;
}

// ─── Main-line API: consume a GPUBuffer directly (Tier A zero-copy) ───
//
// The GPUBuffer must already live on `_capturedDevice` — Rust extracts it via
// wgpu HAL escape hatch (`extract_web_gpu_buffer`). Returns the raw model
// output tensor as Float32Array; postprocess (letterbox inverse + NMS) happens
// Rust-side using LetterboxParams::compute (which the caller computed locally).
export async function ort_run_gpu_buffer(gpu_buffer, dims) {
    if (!_session) throw new Error('ORT session not initialized');
    const tensor = globalThis.ort.Tensor.fromGpuBuffer(gpu_buffer, {
        dataType: 'float32',
        dims: Array.from(dims),
    });
    const result = await _session.run({ [_inputName]: tensor });
    return await result[_outputName].getData();
}

// ─── Fallback API: consume a CPU Float32Array ───
//
// Used when the wgpu HAL escape hatch is unavailable, or when ORT's
// fromGpuBuffer path is known-broken on the target ORT version (§4.5 fallback).
export async function ort_run_cpu_slice(nchw, dims) {
    if (!_session) throw new Error('ORT session not initialized');
    const tensor = new globalThis.ort.Tensor('float32', nchw, Array.from(dims));
    const result = await _session.run({ [_inputName]: tensor });
    return await result[_outputName].getData();
}

// ─── Transition preprocess pipeline (canvas-path only) ───
//
// Only used by ort_detect(canvas). Structurally identical to the Rust
// PreprocessPipeline; shares the same WGSL (injected by build.rs) and the
// same letterbox math (`_computeLetterbox` below vs LetterboxParams::compute).
const DST_SIZE = __DST_SIZE__;
const PREPROCESS_WGSL = `__PREPROCESS_WGSL__`;

let _cvPipeline    = null;
let _cvOutBuf      = null;
let _cvUniBuf      = null;
let _cvInputTex    = null;
let _cvInputTexView = null;
let _cvSampler     = null;

function _ensureCanvasPipeline(device) {
    if (_cvPipeline) return;
    const module = device.createShaderModule({ code: PREPROCESS_WGSL });
    _cvPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });
    _cvOutBuf = device.createBuffer({
        size:  3 * DST_SIZE * DST_SIZE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    _cvUniBuf = device.createBuffer({
        size:  32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cvSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    // NOTE: `_cvInputTex` is allocated lazily on the first useGpu call
    // once we know the actual source aspect (canvas may be any size),
    // and RECREATED only if the aspect ever changes. See the useGpu block.
}

// MUST match Rust `LetterboxParams::compute` exactly (see preprocess.rs tests).
function _computeLetterbox(srcW, srcH, dstSize) {
    const scale = Math.min(dstSize / srcW, dstSize / srcH);
    const padX  = (1 - (srcW * scale) / dstSize) / 2;
    const padY  = (1 - (srcH * scale) / dstSize) / 2;
    return { scale, padX, padY };
}

// ─── Transition API: canvas frame source (open_quartz, external debug tools) ───
//
// Returns a dict `{ output, scale, padX, padY, srcW, srcH }` where:
// - output          — Float32Array, raw model output tensor
// - scale           — letterbox scale factor (matches LetterboxParams.scale)
// - padX, padY      — letterbox pad in DESTINATION-PIXEL units (matches
//                     LetterboxParams::pad_x_px / pad_y_px); feed directly to
//                     `decode_yolo_output` in postprocess.rs.
// - srcW, srcH      — canvas dimensions passed through for postprocess
//
// Main-line callers MUST NOT use this (r4 §15 D10 constraint 1). Rust-side
// consumers are expected to prefer ort_run_gpu_buffer + local LetterboxParams.
// Tier C CPU sub-path — shared by (1) the `useGpu === false` default when the
// device-sharing / fromGpuBuffer preconditions aren't met, (2) the canvas
// backing-tex race fallback (copyExternalImageToTexture async validation
// error), and (3) the ORT fromGpuBuffer catch. `OffscreenCanvas.drawImage`
// runs on a freshly-snapshot canvas so it doesn't share the race hazard of
// the GPU sub-path (2D drawImage silently blocks on a valid tex if needed).
async function _runCpuSubpath(canvas, srcW, srcH, scale) {
    const off = new OffscreenCanvas(DST_SIZE, DST_SIZE);
    const ctx = off.getContext('2d');
    const dw = srcW * scale, dh = srcH * scale;
    const dx = (DST_SIZE - dw) / 2, dy = (DST_SIZE - dh) / 2;
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, DST_SIZE, DST_SIZE);
    ctx.drawImage(canvas, 0, 0, srcW, srcH, dx, dy, dw, dh);
    const imageData = ctx.getImageData(0, 0, DST_SIZE, DST_SIZE);
    const data = new Float32Array(3 * DST_SIZE * DST_SIZE);
    const hw = DST_SIZE * DST_SIZE;
    for (let i = 0; i < hw; i++) {
        data[i]           = imageData.data[i * 4]     / 255;
        data[hw + i]      = imageData.data[i * 4 + 1] / 255;
        data[2 * hw + i]  = imageData.data[i * 4 + 2] / 255;
    }
    const tensor = new globalThis.ort.Tensor('float32', data, [1, 3, DST_SIZE, DST_SIZE]);
    const result = await _session.run({ [_inputName]: tensor });
    const output = await result[_outputName].getData();
    return { output, scale, padX: dx, padY: dy, srcW, srcH };
}

export async function ort_detect(canvas) {
    if (!_session) throw new Error('ORT session not initialized');
    const srcW = canvas.width, srcH = canvas.height;
    const { scale, padX, padY } = _computeLetterbox(srcW, srcH, DST_SIZE);

    // GPU sub-path enablement gate.
    //
    // History (r5-r9, 2026-07-11 to 2026-07-13): earlier revisions concluded
    // this path was blocked by ORT issue #26107 (device-sharing broken).
    // That analysis was wrong. `fromGpuBuffer` + shared `_capturedDevice`
    // work correctly on ORT 1.27 — the probe below proves it end-to-end
    // (writes 0.5 into a GPUBuffer we own, runs session.run against it,
    // reads nonzero output). Earlier all-zero output was two consecutive
    // preprocess bugs:
    //   (1) mixed f32/u32 uniform written via Float32Array only, so
    //       `dst_size=640` landed as f32 bit-pattern 1142947840 in the
    //       shader and the entire index math corrupted (r7 fix).
    //   (2) `copyExternalImageToTexture(src, dst, size)` doesn't scale —
    //       `size` is a 1:1 pixel count copied from the source's origin.
    //       Passing a 4096×3072 canvas + `[640, 480]` copied only the
    //       top-left 640×480 pixels. Fixed with `createImageBitmap(canvas,
    //       { resizeWidth, resizeHeight, resizeQuality: 'high' })` (r9).
    //
    // Preconditions still gated dynamically so we fall back to CPU on:
    // - device sharing check failed (some browsers/adapters can't share)
    // - probe failed (indicates the ORT build in use lacks fromGpuBuffer
    //   or the runtime rejected the buffer for a shape/usage reason)
    // - ort global missing Tensor.fromGpuBuffer (old ORT version)
    //
    // Steady-state at 4096×3072 on RTX 3050 Ti: ~30ms session.run,
    // ~40-50ms end-to-end wall time (vs CPU sub-path ~50-70ms).
    const useGpu = !!(_deviceShared
        && globalThis.__rimeflow_debug_device
        && globalThis.__rimeflow_debug_device.sameObject
        && globalThis.__rimeflow_debug_device.probe
        && globalThis.__rimeflow_debug_device.probe.verdict === 'PASS (fromGpuBuffer works)'
        && globalThis.ort && globalThis.ort.Tensor && globalThis.ort.Tensor.fromGpuBuffer);

    if (useGpu) {
        const device = _capturedDevice;
        _ensureCanvasPipeline(device);

        // 1. Downsample canvas to (640 × 640·srcH/srcW) — GPU-side via
        //    `copyExternalImageToTexture` with `dst.size` smaller than src.
        //    Browser runs bilinear filtering on the GPU. No CPU roundtrip,
        //    no ArrayBuffer allocation, no getImageData sync flush. Reuses
        //    the persistent `_cvInputTex` (only recreated on aspect change).
        //
        //    Aspect preserved with long side = 640. YOLOv8n's fixed
        //    [1,3,640,640] tensor is filled by the compute shader below
        //    which pads letterbox rows/cols with gray around the sampled
        //    region.
        const scaledW = Math.max(1, Math.round(srcW * scale));
        const scaledH = Math.max(1, Math.round(srcH * scale));
        if (!_cvInputTex || _cvInputTex.width !== scaledW || _cvInputTex.height !== scaledH) {
            if (_cvInputTex) _cvInputTex.destroy();
            _cvInputTex = device.createTexture({
                size:   [scaledW, scaledH],
                format: 'rgba8unorm',
                // RENDER_ATTACHMENT is required by copyExternalImageToTexture
                // (WebGPU spec §5.2) — the browser implements the copy as a
                // GPU render pass that downsamples/converts the external image.
                // TEXTURE_BINDING lets the compute shader sample it.
                usage:  GPUTextureUsage.TEXTURE_BINDING
                      | GPUTextureUsage.COPY_DST
                      | GPUTextureUsage.RENDER_ATTACHMENT,
            });
            _cvInputTexView = _cvInputTex.createView();
        }
        // GPU-side downscale. Browser handles bilinear + colorspace conversion
        // in a render pass. No CPU roundtrip.
        //
        // Canvas backing textures live only inside their producer's task
        // (compositor render). If tracking runs between compositor submit and
        // present, `canvas`'s current tex is Invalid and the copy fires an
        // async validation error which ORT surfaces on _its_ next queue op.
        // We push a scope, drop the frame silently on failure — next frame
        // typically catches a fresh canvas tex.
        // Snapshot the canvas ONCE per call, and downscale IN THE SNAPSHOT
        // — `copyExternalImageToTexture` does NOT scale (WebGPU §12.6.4:
        // "size" is a pixel count copied 1:1 from source's origin). If we
        // hand it a 4096×3072 canvas and ask for 640×480 destination it
        // silently keeps the top-left 640×480 pixels of the source.
        // `createImageBitmap` with `resizeWidth/resizeHeight` runs the
        // downscale inside the browser's GPU compositor (bilinear/box), no
        // CPU roundtrip, no ArrayBuffer.
        const frame = await createImageBitmap(canvas, {
            resizeWidth:   scaledW,
            resizeHeight:  scaledH,
            resizeQuality: 'high',
        });
        device.pushErrorScope('validation');
        device.queue.copyExternalImageToTexture(
            { source: frame },
            { texture: _cvInputTex },
            [scaledW, scaledH],
        );
        const _copyErr = await device.popErrorScope();
        if (_copyErr) {
            if (!globalThis.__rimeflow_debug_copy_race_logged) {
                globalThis.__rimeflow_debug_copy_race_logged = true;
                console.warn(
                    '[ort-bridge] canvas copyExternalImageToTexture races with compositor present '
                    + '— falling back to CPU sub-path for affected frames. First error:',
                    _copyErr.message,
                );
            }
            // Bypass the compute pass (its input tex is now stale/invalid) and
            // run the CPU sub-path against a fresh 2D snapshot of the canvas.
            // Costs ~30ms on the race frame; steady-state race rate is low so
            // amortized latency stays close to the pure-GPU path.
            //
            // Alternative: `return null` to let the caller advance its tracker
            // via Kalman-only prediction (see `rimeflow-tracking::wasm_yolo`
            // + `ByteTracker::predict_only`). That is faster but drops one
            // actual detection cycle. Pick per-app: CPU-fallback prioritises
            // detection quality (RimeCut editor preview), null-return
            // prioritises latency (real-time overlay).
            return await _runCpuSubpath(canvas, srcW, srcH, scale);
        }

        // 3. Upload preprocess uniforms.
        //    Shader samples the (scaledW × scaledH) input tex via
        //    normalized (in_u, in_v) ∈ [0, 1] and adds letterbox padding
        //    to reach 640×640 output. src_w/src_h are informational only
        //    (shader uses per-axis region_u/region_v derived from pad_x/y).
        //    WGSL `Params` is mixed f32/u32; write via shared ArrayBuffer
        //    so 640 lands as u32, not f32 bit-pattern (r7 bug we fixed).
        const uniAb   = new ArrayBuffer(32);
        const uniF32  = new Float32Array(uniAb);
        const uniU32  = new Uint32Array(uniAb);
        uniF32[0] = scaledW;      // f32 src_w — uploaded size
        uniF32[1] = scaledH;      // f32 src_h — uploaded size
        uniU32[2] = DST_SIZE;     // u32 dst_size
        uniU32[3] = 0;
        uniF32[4] = scale;        // f32 scale (postprocess mapping)
        uniF32[5] = padX;         // f32 pad_x (normalized)
        uniF32[6] = padY;         // f32 pad_y (normalized)
        uniF32[7] = 0;
        device.queue.writeBuffer(_cvUniBuf, 0, uniAb);

        // 4. Dispatch preprocess compute pass. Persistent sampler + input
        //    view reused. Bind group is cheap to recreate each frame (wgpu
        //    interns identical descriptors) but we could hoist it too.
        const bg = device.createBindGroup({
            layout: _cvPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: _cvInputTexView },
                { binding: 1, resource: _cvSampler },
                { binding: 2, resource: { buffer: _cvOutBuf } },
                { binding: 3, resource: { buffer: _cvUniBuf } },
            ],
        });
        const enc  = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(_cvPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(DST_SIZE / 16), Math.ceil(DST_SIZE / 16));
        pass.end();
        device.queue.submit([enc.finish()]);
        // Do not destroy(tex) synchronously — the compute pass runs async.
        // ORT's session.run below acts as an implicit sync point (it waits
        // for our submit to drain before it feeds the tensor). Let GC or
        // the next iteration's texture recreation reclaim it.

        // 5. Hand ORT the GPUBuffer directly — same device, no cross-device
        //    reference. session.run runs on the shared device, so the whole
        //    GPU pipeline stays local from preprocess through inference.

        try {
            const t_pre  = performance.now();
            const tensor = globalThis.ort.Tensor.fromGpuBuffer(_cvOutBuf, {
                dataType: 'float32',
                dims: [1, 3, DST_SIZE, DST_SIZE],
            });
            const t_run  = performance.now();
            const result = await _session.run({ [_inputName]: tensor });
            const t_data = performance.now();
            const output = await result[_outputName].getData();
            const t_end  = performance.now();
            // 10-frame rolling window session.run + getData timing.
            // Only meaningful on the GPU sub-path (fromGpuBuffer + WebGPU EP);
            // when GPU path is disabled this is unreachable dead code, and
            // when we re-enable GPU path we'll want the numbers back.
            globalThis.__rimeflow_run_timings ??= { n: 0, run: 0, data: 0 };
            const tt = globalThis.__rimeflow_run_timings;
            tt.run  += t_data - t_run;
            tt.data += t_end  - t_data;
            tt.n    += 1;
            if (tt.n === 10) {
                console.log(
                    `[ort-bridge][diag] over ${tt.n} frames  session.run avg=${(tt.run/tt.n).toFixed(1)}ms  getData avg=${(tt.data/tt.n).toFixed(1)}ms`,
                );
                tt.n = 0; tt.run = 0; tt.data = 0;
            }
            // Return original scale/padX/padY (in pixel units) so
            // postprocess reverse-mapping works against the original canvas
            // coordinates. Even though we pre-letterboxed the input, the
            // model output boxes are in DST_SIZE space, and postprocess
            // needs to map them back to source pixels.
            return { output, scale, padX: padX * DST_SIZE, padY: padY * DST_SIZE, srcW, srcH };
        } catch (e) {
            console.warn('[rimeflow-yolov8n] fromGpuBuffer path failed, falling back to CPU tensor:', e);
            // fall through to CPU sub-path
        }
    }

    return await _runCpuSubpath(canvas, srcW, srcH, scale);
}
