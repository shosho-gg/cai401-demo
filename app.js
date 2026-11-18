"use strict";
// CAI401 tool-wear demo. Everything below runs in the browser:
// raw sensor rows -> feedrate-conditioned robust scaling -> multi-scale feature images -> ONNX autoencoder
// -> per-pair dynamic thresholds + motor-load monitor -> ensemble -> causal smoothing -> decision.

const $ = id => document.getElementById(id);
const S = { meta: null, sensors: null, mu: null, sd: null, session: null, N: 0, exps: {}, busy: false };
const COL_NORMAL = "#8DB97A", COL_WORN = "#1E261E", COL_FLAG = "#B3261E", COL_OK = "#B9C6B3", COL_DARK = "#2C3D2C";
let current = 8, playing = false, playToken = 0;

// ------------------------------------------------------------------ loading
async function init() {
  try {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL("./", location.href).href;
    // binary files are shipped as base64 inside JSON (the host serves .json, not .bin/.onnx)
    const b64 = o => { const bin = atob(o.data), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
    const [meta, sens, pairs] = await Promise.all([
      fetch("meta.json").then(r => r.json()),
      fetch("sensors.json").then(r => r.json()).then(o => b64(o).buffer),
      fetch("pairstats.json").then(r => r.json()).then(o => b64(o).buffer)]);
    S.meta = meta; S.N = meta.cols.length; S.sensors = new Float32Array(sens);
    const p = new Float32Array(pairs), K = p.length / 2; S.mu = p.subarray(0, K); S.sd = p.subarray(K);
    buildStatic();
    $("engine").textContent = "Loading neural network…";
    const modelBytes = await fetch("model.json").then(r => r.json()).then(b64);
    S.session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
    $("engine").textContent = `✓ Neural network loaded (ONNX, ${S.N} sensors, WebAssembly). Every score on this page is computed live in your browser.`;
    $("engine").style.display = "none";
    ["play", "reset", "analyze", "runall"].forEach(id => $(id).disabled = false);
  } catch (err) {
    console.error(err && err.stack);
    $("engine").textContent = "Could not load the model in this browser: " + err;
  }
}

// ------------------------------------------------------------------ data per experiment
function exp(e) {
  if (S.exps[e]) return S.exps[e];
  const d = S.meta.experiments[e], N = S.N, n = d.n_rows, clip = S.meta.clip;
  const raw = S.sensors.subarray(d.offset, d.offset + n * N), xs = new Float32Array(n * N);
  for (let t = 0; t < n; t++) for (let j = 0; j < N; j++) {
    let v = (raw[t * N + j] - d.center[j]) / d.scale[j];
    xs[t * N + j] = v > clip ? clip : v < -clip ? -clip : v;
  }
  return (S.exps[e] = { e, d, raw, xs, frames: new Map(), res: [], ens: [] });
}

function frame(E, t) {                       // signature matrices for windows 10/30/60 ending at sample t
  if (E.frames.has(t)) return E.frames.get(t);
  const N = S.N, W = S.meta.windows, out = new Float32Array(W.length * N * N), xs = E.xs;
  W.forEach((w, c) => {
    const base = c * N * N;
    for (let k = t - w + 1; k <= t; k++) {
      const o = k * N;
      for (let i = 0; i < N; i++) {
        const xi = xs[o + i]; if (xi === 0) continue;
        const bi = base + i * N;
        for (let j = 0; j < N; j++) out[bi + j] += xi * xs[o + j];
      }
    }
    for (let q = base; q < base + N * N; q++) out[q] /= w;
  });
  E.frames.set(t, out); return out;
}

async function computeSeq(E, q, detail = false) {
  if (E.res[q] && (!detail || E.res[q].detail)) return E.res[q];
  const N = S.N, M = S.meta, hi = E.d.hi[q], F = 3 * N * N;
  const input = new Float32Array(4 * F); let target;
  for (let f = 0; f < 5; f++) {
    const fr = frame(E, hi - (4 - f) * M.stride);
    if (f < 4) input.set(fr, f * F); else target = fr;
  }
  const t0 = performance.now();
  const out = await S.session.run({ frames: new ort.Tensor("float32", input, [1, 4, 3, N, N]) });
  const ms = performance.now() - t0, pred = out.pred.data;
  let cnt = 0; const rowEx = detail ? new Float32Array(N) : null, err = detail ? new Float32Array(F) : null;
  for (let k = 0; k < F; k++) {
    const r = (pred[k] - target[k]) ** 2;
    if (detail) err[k] = r;
    if ((r - S.mu[k]) / S.sd[k] > M.z) { cnt++; if (detail) rowEx[Math.floor((k % (N * N)) / N)]++; }
  }
  const a = cnt / F;
  let p = 0;                                  // motor power / current monitor over the 80 raw samples of the sequence
  for (let k = hi - M.span + 1; k <= hi; k++) { let s = 0; for (const j of M.phys_idx) s += E.xs[k * N + j]; p += s / M.phys_idx.length; }
  p /= M.span;
  const ens = 0.5 * ((a - M.std_a[0]) / M.std_a[1] + (p - M.std_p[0]) / M.std_p[1]);
  E.ens[q] = ens;
  const res = { a, p, ens, ms };
  if (detail) Object.assign(res, { detail: true, target, pred: Float32Array.from(pred), err, rowEx });
  E.res[q] = res; return res;
}

function median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }
async function smoothedScore(E, q) {         // causal rolling median of the last W ensemble scores
  const W = S.meta.smooth_w;
  for (let k = Math.max(0, q - W + 1); k <= q; k++) if (E.ens[k] === undefined) await computeSeq(E, k);
  return median(E.ens.slice(Math.max(0, q - W + 1), q + 1));
}

// ------------------------------------------------------------------ drawing helpers
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 600, h = +cv.getAttribute("height");
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.style.height = h + "px"; }
  const ctx = cv.getContext("2d"); ctx.setTransform(cv.width / w, 0, 0, cv.width / w, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function plot(cv, o) {
  const { ctx, w, h } = setupCanvas(cv), L = 52, R = 24, T = 10, B = 26;
  const xs = o.series.flatMap(s => s.x), ys = o.series.flatMap(s => s.y).concat(o.thr !== undefined ? [o.thr] : []);
  let x0 = o.xmin ?? Math.min(...xs), x1 = o.xmax ?? Math.max(...xs); if (!(x1 > x0)) x1 = x0 + 1;
  let y0 = o.ymin ?? Math.min(...ys), y1 = o.ymax ?? Math.max(...ys); if (!isFinite(y0)) { y0 = 0; y1 = 1; }
  const pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad;
  const X = v => L + (v - x0) / (x1 - x0) * (w - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (h - T - B);
  ctx.font = "11px Inter, Arial"; ctx.fillStyle = "#56635A"; ctx.strokeStyle = "#E1EADC"; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) { const v = y0 + (y1 - y0) * i / 3, y = Y(v); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(fmt(v), L - 6, y + 4); }
  ctx.textAlign = "center"; for (let i = 0; i <= 4; i++) { const v = x0 + (x1 - x0) * i / 4; ctx.fillText(v.toFixed(0) + " s", X(v), h - 8); }
  for (const s of o.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.4; ctx.beginPath(); let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const gap = i > 0 && s.gap && s.x[i] - s.x[i - 1] > s.gap;
      if (!pen || gap) { ctx.moveTo(X(s.x[i]), Y(s.y[i])); pen = true; } else ctx.lineTo(X(s.x[i]), Y(s.y[i]));
    }
    ctx.stroke();
    if (s.flags) for (let i = 0; i < s.x.length; i++) { ctx.fillStyle = s.flags[i] ? COL_FLAG : COL_OK; ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), s.flags[i] ? 3.2 : 2.4, 0, 7); ctx.fill(); }
  }
  if (o.thr !== undefined) { ctx.setLineDash([6, 5]); ctx.strokeStyle = COL_DARK; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(L, Y(o.thr)); ctx.lineTo(w - R, Y(o.thr)); ctx.stroke(); ctx.setLineDash([]); }
}
function fmt(v) { const a = Math.abs(v); return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : a >= 0.01 ? v.toFixed(3) : v.toExponential(1); }
function heat(cv, data, off, N, vmin, vmax, palette) {
  const ctx = cv.getContext("2d"), img = ctx.createImageData(N, N);
  for (let k = 0; k < N * N; k++) {
    const t = Math.max(0, Math.min(1, (data[off + k] - vmin) / (vmax - vmin || 1))), c = palette(t);
    img.data[k * 4] = c[0]; img.data[k * 4 + 1] = c[1]; img.data[k * 4 + 2] = c[2]; img.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
const lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
function ramp(stops) { return t => { const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)); return lerp(stops[i], stops[i + 1], t * n - i); }; }
const PAL_VAL = ramp([[240, 246, 236], [189, 219, 172], [116, 168, 98], [56, 108, 52], [26, 50, 28]]);
const PAL_ERR = ramp([[255, 255, 255], [208, 228, 196], [96, 150, 82], [28, 52, 30]]);

// ------------------------------------------------------------------ static UI
function testIds() { return Object.keys(S.meta.experiments).map(Number).filter(e => !S.meta.experiments[e].train).sort((a, b) => a - b); }
function chip(e, onClick) {
  const d = S.meta.experiments[e], b = document.createElement("button");
  b.className = "chip"; b.type = "button"; b.dataset.e = e;
  b.innerHTML = `Exp ${e}<small>${d.label ? "worn" : "normal"} · f ${d.feedrate}</small>`;
  b.addEventListener("click", () => onClick(e)); return b;
}
function markChips() { document.querySelectorAll("#chips-monitor .chip, #chips-inside .chip").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.e === current))); }
function buildStatic() {
  testIds().forEach(e => { $("chips-monitor").appendChild(chip(e, selectExp)); $("chips-inside").appendChild(chip(e, selectExp)); });
  markChips(); buildResultsTable(); buildPairs(); resetMonitor(); setupInside();
}
function selectExp(e) { stopPlay(); current = e; markChips(); resetMonitor(); setupInside(); }

// ------------------------------------------------------------------ control room
const TRACE_COLS = ["X1_OutputPower", "Y1_OutputPower", "S1_OutputPower", "S1_CurrentFeedback"];
function resetMonitor() {
  const E = exp(current); stopPlay();
  $("light").className = "light"; $("status").innerHTML = "Ready<small>Press Start to stream the experiment</small>";
  $("k-exp").textContent = `${current} · f ${E.d.feedrate}`; $("k-time").textContent = "0 s"; $("k-n").textContent = "0";
  $("k-frac").textContent = "–"; $("k-score").textContent = "–"; $("k-ms").textContent = "–";
  $("truth").innerHTML = "True condition is revealed at the end."; $("play").textContent = "▶ Start";
  drawTraces(E, E.d.hi[0]); plot($("scorechart"), { series: [{ x: [], y: [], color: "#C6D6BD" }], thr: S.meta.threshold, xmin: 0, xmax: 1 });
  E.mon = { q: 0, xs: [], ys: [], fl: [] };
}
function drawTraces(E, hi) {
  const N = S.N, n0 = Math.max(0, hi - 299);
  TRACE_COLS.forEach((c, k) => {
    const j = S.meta.cols.indexOf(c), x = [], y = [];
    for (let t = n0; t <= hi; t++) { x.push(t / S.meta.fs); y.push(E.raw[t * N + j]); }
    plot($("tr" + k), { series: [{ x, y, color: COL_DARK, width: 1.2 }], xmin: Math.max(0, hi / S.meta.fs - 30), xmax: Math.max(30, hi / S.meta.fs) });
  });
}
function stopPlay() { playing = false; playToken++; $("play").textContent = "▶ Start"; }
async function play() {
  if (playing) { stopPlay(); $("play").textContent = "▶ Continue"; return; }
  const E = exp(current), token = ++playToken; playing = true; $("play").textContent = "❚❚ Pause";
  if (E.mon.q === 0) { $("truth").innerHTML = "True condition is revealed at the end."; E.mon.xs = []; E.mon.ys = []; E.mon.fl = []; }
  const M = S.meta, nq = E.d.hi.length;
  while (playing && token === playToken && E.mon.q < nq) {
    const q = E.mon.q, tStart = performance.now();
    const r = await computeSeq(E, q), s = await smoothedScore(E, q), flagged = s > M.threshold;
    if (token !== playToken) return;
    const hi = E.d.hi[q], tsec = hi / M.fs;
    E.mon.xs.push(tsec); E.mon.ys.push(s); E.mon.fl.push(flagged); E.mon.q++;
    const frac = E.mon.fl.filter(Boolean).length / E.mon.fl.length;
    drawTraces(E, hi);
    plot($("scorechart"), { series: [{ x: E.mon.xs, y: E.mon.ys, color: "#C6D6BD", flags: E.mon.fl, gap: 1.5 }], thr: M.threshold,
      xmin: E.d.hi[0] / M.fs, xmax: E.d.hi[nq - 1] / M.fs });
    const alarm = frac > M.run_flag_frac;
    $("light").className = "light " + (alarm ? "red" : flagged ? "amber" : "green");
    $("status").innerHTML = alarm ? "ALARM: tool wear suspected<small>More than 10% of sequences flagged</small>"
      : flagged ? "Anomalous sequence<small>Above threshold; watching for persistence</small>" : "Normal operation<small>Score below threshold</small>";
    $("k-time").textContent = `${(E.mon.q * 0.5).toFixed(1)} s`; $("k-n").textContent = `${E.mon.q} / ${nq}`;
    $("k-frac").textContent = `${(100 * frac).toFixed(1)}%`; $("k-score").textContent = `${s.toFixed(2)} / ${M.threshold.toFixed(2)}`;
    $("k-ms").textContent = `${r.ms.toFixed(0)} ms`;
    const wait = 500 / +$("speed").value - (performance.now() - tStart);
    await new Promise(res => setTimeout(res, Math.max(0, wait)));
  }
  if (E.mon.q >= nq) {
    playing = false; $("play").textContent = "▶ Start";
    const frac = E.mon.fl.filter(Boolean).length / nq, predW = frac > M.run_flag_frac, truthW = !!E.d.label;
    $("truth").innerHTML = `<b>Finished.</b> Prediction: <b>${predW ? "WORN" : "NORMAL"}</b> · True condition: <b>${truthW ? "WORN" : "NORMAL"}</b> (visual inspection: ${E.d.inspection}) · ${predW === truthW ? "✓ correct" : "✗ wrong"}`;
    E.mon.q = 0; E.mon.xs = []; E.mon.ys = []; E.mon.fl = [];
  }
}

// ------------------------------------------------------------------ inside the model
function setupInside() {
  const E = exp(current), sl = $("seqslider"); sl.max = E.d.hi.length - 1;
  if (+sl.value > +sl.max) sl.value = 0;
  if (E.mon && E.mon.q > 0) sl.value = E.mon.q - 1;
  updateSeqInfo();
}
function updateSeqInfo() { const E = exp(current), q = +$("seqslider").value; $("seqinfo").textContent = `Exp ${current}, sequence ${q + 1} of ${E.d.hi.length} (t = ${(E.d.hi[q] / S.meta.fs).toFixed(1)} s)`; }
async function analyze() {
  if (!S.session) return;
  const E = exp(current), q = +$("seqslider").value, N = S.N, NN = N * N;
  $("analyze").disabled = true; $("analyze").textContent = "Running…";
  const r = await computeSeq(E, q, true), s = await smoothedScore(E, q);
  for (let c = 0; c < 3; c++) {
    let lo = Infinity, hi = -Infinity;
    for (let k = c * NN; k < (c + 1) * NN; k++) { lo = Math.min(lo, r.target[k], r.pred[k]); hi = Math.max(hi, r.target[k], r.pred[k]); }
    heat($("m" + c + "0"), r.target, c * NN, N, lo, hi, PAL_VAL);
    heat($("m" + c + "1"), r.pred, c * NN, N, lo, hi, PAL_VAL);
    let em = 0; for (let k = c * NN; k < (c + 1) * NN; k++) em = Math.max(em, r.err[k]);
    heat($("m" + c + "2"), r.err, c * NN, N, 0, em, PAL_ERR);
  }
  const share = Array.from(r.rowEx).map((v, i) => ({ c: S.meta.cols[i], v: v / (3 * N) })).sort((a, b) => b.v - a.v).slice(0, 8);
  const mx = Math.max(0.01, share[0].v);
  $("sensorbars").innerHTML = share.map(o => `<div class="meter"><span>${o.c}</span><div class="track"><div class="fill" style="width:${(100 * o.v / mx).toFixed(1)}%; background:${COL_FLAG}"></div></div><span>${(100 * o.v).toFixed(1)}%</span></div>`).join("");
  $("i-a").textContent = `${(100 * r.a).toFixed(2)}%`;
  $("i-s").innerHTML = `${s.toFixed(2)} <span class="note">${s > S.meta.threshold ? "flagged" : "below threshold"}</span>`;
  $("mapnote").textContent = `Exp ${current} at t = ${(E.d.hi[q] / S.meta.fs).toFixed(1)} s. Rows and columns are the 39 sensors in the order X axis, Y axis, Z axis, spindle. Network inference: ${r.ms.toFixed(0)} ms.`;
  $("analyze").disabled = false; $("analyze").textContent = "Run the network on this moment";
}

// ------------------------------------------------------------------ normal vs worn
const PAIRS = [{ f: 20, n: 2, w: 8 }, { f: 6, n: 4, w: 6 }, { f: 3, n: 12, w: 13 }];
function buildPairs() {
  PAIRS.forEach((p, i) => {
    const b = document.createElement("button"); b.className = "chip"; b.type = "button";
    b.textContent = `Feedrate ${p.f}: normal ${p.n} vs worn ${p.w}`; b.addEventListener("click", () => showPair(i)); b.dataset.i = i;
    $("chips-pairs").appendChild(b);
  });
  showPair(0);
}
function cuttingSeries(e, cols) {
  const E = exp(e), N = S.N, js = cols.map(c => S.meta.cols.indexOf(c)), x = [], y = [];
  let k = 0, acc = [];
  for (let t = 0; t < E.d.n_rows; t++) if (E.d.cutting[t]) {
    let v = 0; for (const j of js) v += Math.abs(E.raw[t * N + j]);
    acc.push(v); if (acc.length > 10) acc.shift();
    x.push(k / S.meta.fs); y.push(acc.reduce((a, b) => a + b, 0) / acc.length); k++;
  }
  return { x, y };
}
function meanAbs(e, c) { const E = exp(e), N = S.N, j = S.meta.cols.indexOf(c); let s = 0, n = 0; for (let t = 0; t < E.d.n_rows; t++) if (E.d.cutting[t]) { s += Math.abs(E.raw[t * N + j]); n++; } return s / n; }
function verdictText(e) {
  const d = S.meta.experiments[e];
  if (d.train) return `normal ${e} was a training experiment (model not tested on it)`;
  return `exp ${e}: ${(100 * d.ref_frac).toFixed(1)}% of sequences flagged → ${d.ref_frac > S.meta.run_flag_frac ? "WORN" : "NORMAL"}`;
}
function showPair(i) {
  document.querySelectorAll("#chips-pairs .chip").forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.i === i)));
  const p = PAIRS[i], feed = ["X1_OutputPower", "Y1_OutputPower"];
  const a = cuttingSeries(p.n, feed), b = cuttingSeries(p.w, feed);
  $("cmp-title").textContent = `Feed-axis motor power (X + Y) during cutting, feedrate ${p.f}`;
  $("lg-n").textContent = `normal experiment ${p.n}`; $("lg-w").textContent = `worn experiment ${p.w}`;
  plot($("cmpchart"), { series: [{ x: a.x, y: a.y, color: COL_NORMAL, width: 1.6 }, { x: b.x, y: b.y, color: COL_WORN, width: 1.6 }], ymin: 0 });
  const rows = [["X-axis power", "X1_OutputPower"], ["Y-axis power", "Y1_OutputPower"], ["Spindle power", "S1_OutputPower"], ["Spindle current", "S1_CurrentFeedback"]];
  $("meters").innerHTML = rows.map(([lab, c]) => {
    const vn = meanAbs(p.n, c), vw = meanAbs(p.w, c), m = Math.max(vn, vw) || 1;
    return `<div style="margin-bottom:8px"><b style="font-size:13px;color:#2C3D2C">${lab}</b>
      <div class="meter"><span>normal ${p.n}</span><div class="track"><div class="fill" style="width:${(100 * vn / m).toFixed(1)}%;background:${COL_NORMAL}"></div></div><span>${fmt(vn)}</span></div>
      <div class="meter"><span>worn ${p.w}</span><div class="track"><div class="fill" style="width:${(100 * vw / m).toFixed(1)}%;background:${COL_WORN}"></div></div><span>${fmt(vw)}</span></div></div>`;
  }).join("");
  const rn = meanAbs(p.n, "X1_OutputPower") + meanAbs(p.n, "Y1_OutputPower"), rw = meanAbs(p.w, "X1_OutputPower") + meanAbs(p.w, "Y1_OutputPower"), ratio = rw / rn;
  $("cmp-text").innerHTML = `Feed-axis power of the worn tool is <b>${ratio.toFixed(2)}×</b> that of the normal tool at feedrate ${p.f}. ` +
    (ratio > 1.4 ? "The extra cutting force is clearly visible, so this wear is detectable." : "The difference is small: in the sensor data this worn tool looks almost like a normal one.") +
    `<br><span style="font-size:13px">Model: ${verdictText(p.w)}; ${verdictText(p.n)}.</span>`;
}

// ------------------------------------------------------------------ all results
function buildResultsTable(browser = {}) {
  const M = S.meta, rows = testIds().map(e => {
    const d = M.experiments[e], b = browser[e], frac = b ? b.frac : d.ref_frac, alarm = frac > M.run_flag_frac;
    return { e, d, frac, alarm, ok: alarm === !!d.label };
  });
  const groups = [
    { t: "Normal tools", sub: "", f: r => !r.d.label },
    { t: "Worn tools · high feedrate (15–20)", sub: "", f: r => r.d.label && r.d.feedrate >= 15 },
    { t: "Worn tools · low feedrate (3–12)", sub: "", f: r => r.d.label && r.d.feedrate < 15 }];
  const G = groups.map(g => ({ ...g, rows: g.f ? rows.filter(g.f).sort((x, y) => y.d.feedrate - x.d.feedrate || x.e - y.e) : [] }));
  const nOK = G[0].rows.filter(r => r.ok).length, nHi = G[1].rows.filter(r => r.alarm).length, nLo = G[2].rows.filter(r => r.alarm).length;
  $("sumcards").innerHTML = `
    <div class="sum"><div class="big">${nOK}<small> / ${G[0].rows.length}</small></div><div class="t">Normal tools correctly identified</div></div>
    <div class="sum red"><div class="big">${nHi}<small> / ${G[1].rows.length}</small></div><div class="t">Worn tools detected · high feedrate</div></div>
    <div class="sum dim"><div class="big">${nLo}<small> / ${G[2].rows.length}</small></div><div class="t">Worn tools detected · low feedrate</div></div>`;
  const insp = v => v === "yes" ? "passed visual inspection" : v === "no" ? "failed visual inspection" : "not inspected";
  $("groups").innerHTML = G.map((g, gi) => `<div class="grp"><div class="grp-h"><h3>${g.t}</h3></div>` + g.rows.map(r => {
    const pill = !r.d.label ? (r.alarm ? '<span class="pill fa">false alarm</span>' : '<span class="pill quiet">✓ normal</span>')
      : (r.alarm ? '<span class="pill hit">✓ worn detected</span>' : '<span class="pill miss">✗ missed</span>');
    return `<div class="row" data-e="${r.e}"><div class="ex"><b>Experiment ${r.e}</b><small>feedrate ${r.d.feedrate}</small></div>
      <div class="track" title="share of sequences flagged; dashed line = 10% alarm rule"><div class="fill ${r.alarm ? "" : "low"}" style="width:${Math.max(0.6, 100 * r.frac).toFixed(1)}%"></div><div class="mark"></div></div>
      <div class="pct">${(100 * r.frac).toFixed(1)}%</div>${pill}</div>`;
  }).join("") + `</div>`).join("");
  document.querySelectorAll("#groups .row").forEach(el => el.addEventListener("click", () => { selectExp(+el.dataset.e); location.hash = "#monitor"; }));
  const correct = rows.filter(r => r.ok).length, ran = Object.keys(browser).length;
  const nn = G[0].rows.length, nw = G[1].rows.length + G[2].rows.length, ba = (nOK / nn + (nHi + nLo) / nw) / 2;
  $("resfoot").innerHTML = `Bar = % of sequences flagged · dashed line = 10% alarm rule · ${correct} of ${rows.length} experiments correct · click a row to watch it.`;
}
async function runAll() {
  if (S.busy) return; S.busy = true; stopPlay(); $("runall").disabled = true;
  const ids = testIds(), total = ids.reduce((s, e) => s + S.meta.experiments[e].hi.length, 0), out = {}; let done = 0; const t0 = performance.now();
  for (const e of ids) {
    const E = exp(e), d = E.d; let fl = 0, maxdiff = 0;
    for (let q = 0; q < d.hi.length; q++) {
      await computeSeq(E, q); const s = await smoothedScore(E, q);
      if (s > S.meta.threshold) fl++; maxdiff = Math.max(maxdiff, Math.abs(s - d.ref_score[q]));
      done++; if (done % 8 === 0) { $("prog").style.width = (100 * done / total).toFixed(1) + "%"; $("runinfo").textContent = `${done} / ${total} sequences · exp ${e}`; await new Promise(r => setTimeout(r, 0)); }
    }
    out[e] = { frac: fl / d.hi.length, maxdiff }; buildResultsTable(out);
  }
  $("prog").style.width = "100%";
  const worst = Math.max(...Object.values(out).map(o => o.maxdiff));
  $("runinfo").textContent = `Done: ${total} sequences in ${((performance.now() - t0) / 1000).toFixed(0)} s. Largest difference to Python: ${worst.toExponential(1)}.`;
  $("runall").disabled = false; S.busy = false;
}

// ------------------------------------------------------------------ routing and events
function route() {
  const v = (location.hash || "#monitor").slice(1), ok = ["monitor", "inside", "compare", "results"].includes(v) ? v : "monitor";
  document.querySelectorAll(".view").forEach(s => s.classList.toggle("on", s.id === "v-" + ok));
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.dataset.v === ok));
  if (S.meta) { if (ok === "inside") setupInside(); if (ok === "compare") showPair(+(document.querySelector('#chips-pairs .chip[aria-pressed="true"]')?.dataset.i || 0)); if (ok === "monitor") drawTraces(exp(current), exp(current).d.hi[Math.max(0, (exp(current).mon?.q || 1) - 1)]); }
}
window.addEventListener("hashchange", route);
$("play").addEventListener("click", play);
$("reset").addEventListener("click", resetMonitor);
$("analyze").addEventListener("click", analyze);
$("runall").addEventListener("click", runAll);
$("seqslider").addEventListener("input", updateSeqInfo);
route(); init();
