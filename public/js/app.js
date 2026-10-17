import { SIGNALS, LAYERS, LIVE_TTL_MS } from './signals.js';
import { CompoundDetector, computeTrust, trustLabel } from './compound.js';
import { VideoMetrics, mouthAperture } from './videoMetrics.js';
import { AudioAnalyzer } from './audioAnalyzer.js';
import { CrossModal } from './crossmodal.js';
import { Vision } from './vision.js';
import { CHALLENGES, pickChallenge } from './gestures.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (n) => Math.round(n * 100);

// ---------- state ----------
const sigState = new Map(); // id -> { fired, severity, detail, t }
const compound = new CompoundDetector({ windowMs: 10000, minSignals: 2 });
const videoMetrics = new VideoMetrics();
const crossModal = new CrossModal();
const audio = new AudioAnalyzer({ onEnvelope: (t, v) => crossModal.pushAudio(t, v) });
const vision = new Vision();

let config = { scenarios: [] };
let stream = null;
let visionTimer = null;
let currentDemo = '';
let geo = null; // { lat, lon }
let marker = null;
let markerUntil = 0;
let scam = null;
let challenge = null; // { steps, i, state: 'suggested'|'loading'|'running'|'passed'|'failed', hits, stepStart }
let challengeCooldownUntil = 0;
let lastFaceT = 0;
const callStart = performance.now();
let t0 = callStart;

// ---------- log ----------
function log(text, cls = '') {
  const li = document.createElement('li');
  if (cls) li.className = cls;
  const s = Math.max(0, Math.round((performance.now() - t0) / 1000));
  li.innerHTML = `<time>${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}</time>${esc(text)}`;
  const ol = $('#log');
  ol.prepend(li);
  while (ol.children.length > 60) ol.lastChild.remove();
}

// ---------- signals ----------
function report(id, r, now = performance.now()) {
  if (!SIGNALS[id]) return;
  const prev = sigState.get(id);
  const fired = Boolean(r.fired);
  const severity = fired ? Number(r.severity) || 0.5 : 0;
  sigState.set(id, { fired, severity, detail: r.detail || '', t: now });
  compound.report(id, { fired, severity }, now);
  if (fired && !prev?.fired) log(`${SIGNALS[id].label}: ${r.detail}`);
}

function activeSignals() {
  return [...sigState].filter(([, s]) => s.fired).map(([id, s]) => ({ id, severity: s.severity }));
}

// ---------- rendering ----------
function render(now) {
  const markerActive = marker && now < markerUntil;
  const trust = computeTrust(activeSignals(), markerActive ? marker.strength : 0);
  const lab = trustLabel(trust);
  $('#trustPct').textContent = pct(trust);
  $('#trustLabel').textContent = sigState.size || stream ? lab.text : 'Waiting for a call';
  const tr = $('.trust');
  tr.classList.remove('ok', 'warn', 'bad');
  tr.classList.add(lab.key);
  $('#ringFg').style.strokeDashoffset = String(264 * (1 - trust));

  const alert = $('#alert');
  if (markerActive) {
    alert.hidden = false;
    alert.innerHTML =
      `<b>Compound warning</b> &middot; ${marker.signals.length} independent signals together` +
      `<div class="chips">${marker.signals.map((id) => `<span class="chip live">${esc(SIGNALS[id].label)}</span>`).join('')}` +
      `${marker.standing.map((id) => `<span class="chip">${esc(SIGNALS[id].label)}</span>`).join('')}</div>` +
      (marker.synergy.length ? `<div class="hint" style="margin:6px 0 0">Reinforcing pair: ${esc(marker.synergy.join(', '))}</div>` : '');
  } else {
    alert.hidden = true;
  }

  $('#groups').innerHTML = Object.entries(LAYERS).map(([layer, meta]) => {
    const rows = Object.entries(SIGNALS).filter(([, d]) => d.layer === layer).map(([id, def]) => {
      const s = sigState.get(id);
      const cls = !s ? 'idle' : s.fired ? (s.severity >= 0.6 ? 'bad' : 'warn') : 'ok';
      const detail = s?.detail ? `<div class="detail">${esc(s.detail)}</div>` : '';
      return `<div class="sig ${cls}"><span class="led"></span><div><div class="name">${esc(def.label)}</div>${detail}</div></div>`;
    }).join('');
    return `<div class="group"><h4>${esc(meta.label)} &middot; ${esc(meta.question)}</h4>${rows}</div>`;
  }).join('');

  renderScam();
  renderChallenge(trust, now);
}

function renderScam() {
  const el = $('#scamCard');
  if (!scam) { el.hidden = true; return; }
  el.hidden = false;
  const bad = scam.category !== 'none';
  el.className = `scam ${bad ? 'bad' : ''}`;
  const flags = scam.red_flags.map((f) => `<li><b>${esc(f.tactic.replace(/_/g, ' '))}</b>${f.quote ? ` <q>${esc(f.quote)}</q>` : ''}<br><span class="hint" style="margin:0">${esc(f.why)}</span></li>`).join('');
  const acts = scam.recommended_actions.map((a) => `<li>${esc(a)}</li>`).join('');
  el.innerHTML =
    `<h3>Content analysis &middot; ${esc(scam.engine)}</h3>` +
    `<div class="cat">${bad ? esc(scam.category.replace(/_/g, ' ')) : 'No scam pattern'} <span class="hint" style="margin:0">${pct(scam.confidence)}% confidence</span></div>` +
    `<div>${esc(scam.summary)}</div>` +
    (flags ? `<ul>${flags}</ul>` : '') +
    (acts ? `<h3 style="margin-top:10px">What to do</h3><ul class="actions">${acts}</ul>` : '') +
    (scam.note ? `<div class="hint">${esc(scam.note)}</div>` : '');
}

// ---------- tick ----------
function tick() {
  const now = performance.now();
  if (stream) {
    for (const r of videoMetrics.evaluate()) report(r.id, r, now);
    for (const r of audio.evaluate()) report(r.id, r, now);
    for (const r of crossModal.evaluate(now)) report(r.id, r, now);
    $('#srcStatus').textContent = now - lastFaceT > 3000 ? 'No face detected in the shared video.' : `Tracking face (${videoMetrics.frames} frames).`;
  }
  for (const [id, s] of sigState) {
    if (!SIGNALS[id].standing && s.fired && now - s.t > LIVE_TTL_MS) report(id, { fired: false }, now);
  }
  const m = compound.evaluate(now);
  if (m) {
    marker = m;
    markerUntil = now + 12000;
    log(`COMPOUND: ${m.signals.map((id) => SIGNALS[id].label).join(' + ')}${m.standing.length ? ` (with ${m.standing.length} standing flags)` : ''}`, 'compound');
  }
  render(now);
}

// ---------- metadata (part 1) ----------
async function api(path, body) {
  const res = await fetch(path, { method: body ? 'POST' : 'GET', headers: { 'content-type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  const j = await res.json();
  if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
  return j;
}

function parseUserLoc() {
  const v = $('#userLoc').value.trim();
  if (!v) return undefined;
  const [country, ...rest] = v.split('/').map((s) => s.trim());
  return { country: country.toUpperCase(), region: rest.join('/') };
}

async function checkNumber() {
  const status = $('#metaStatus');
  status.textContent = 'Checking...';
  try {
    const body = {
      demo: currentDemo || undefined,
      number: $('#number').value.trim(),
      claimedOrg: $('#claimedOrg').value.trim(),
      verstat: $('#verstat').value,
      originCountry: $('#originCountry').value.trim().toUpperCase(),
      claimedLocal: $('#claimedLocal').checked,
      user: parseUserLoc(),
      lat: geo?.lat,
      lon: geo?.lon,
    };
    const r = await api('/api/metadata', body);
    const now = performance.now();
    for (const s of r.signals) report(s.id, s, now);
    status.textContent = `${r.number} · ${r.lineType}${r.carrier ? ' · ' + r.carrier : ''} · source: ${r.source}`;
    if (r.transcript && !$('#transcript').value.trim()) $('#transcript').value = r.transcript;
    if (r.source === 'none') status.textContent += ' (no IPQS key: only line type and attestation were checked)';
    if (!r.signals.some((s) => s.fired)) log('Metadata: no contradictions found.', 'pass');
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
}

function useGeolocation() {
  if (!navigator.geolocation) return ($('#metaStatus').textContent = 'Geolocation is not available.');
  navigator.geolocation.getCurrentPosition(
    (p) => {
      geo = { lat: p.coords.latitude, lon: p.coords.longitude };
      $('#userLoc').placeholder = 'Location captured; server will resolve it';
      $('#metaStatus').textContent = 'Location captured.';
    },
    (e) => ($('#metaStatus').textContent = `Location unavailable: ${e.message}`),
    { timeout: 8000 },
  );
}

// ---------- content (part 4) ----------
async function classifyNow() {
  const transcript = $('#transcript').value.trim();
  const st = $('#contentStatus');
  if (!transcript) return (st.textContent = 'Nothing to analyze yet.');
  st.textContent = 'Analyzing...';
  try {
    scam = await api('/api/classify', { transcript, claimedOrg: $('#claimedOrg').value.trim() });
    const bad = scam.category !== 'none' && scam.confidence >= 0.5;
    report('SCAM_CONTENT', {
      fired: bad, severity: scam.confidence,
      detail: bad ? `${scam.category.replace(/_/g, ' ')} (${pct(scam.confidence)}%): ${scam.summary}` : 'No scam script detected.',
    });
    st.textContent = `Done (${scam.engine}).`;
  } catch (err) {
    st.textContent = `Error: ${err.message}`;
  }
}

let recognition = null;
let lastClassifiedLen = 0;
function toggleSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return ($('#contentStatus').textContent = 'Speech recognition is not supported in this browser. Use Chrome or Edge.');
  if (recognition) {
    recognition.stop();
    recognition = null;
    $('#btnSpeech').textContent = 'Live transcript (microphone)';
    return;
  }
  recognition = new SR();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';
  recognition.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) $('#transcript').value += (($('#transcript').value && ' ') + e.results[i][0].transcript.trim());
    }
    const len = $('#transcript').value.length;
    if (len - lastClassifiedLen > 80) { lastClassifiedLen = len; classifyNow(); }
  };
  recognition.onend = () => { if (recognition) recognition.start(); };
  recognition.start();
  $('#btnSpeech').textContent = 'Stop live transcript';
}

// ---------- capture (part 2 and 3) ----------
async function startCapture(kind, file = null) {
  try {
    stopCapture();
    let srcUrl = null;

    if (kind === 'screen') {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: true });
    } else if (kind === 'cam') {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 }, audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } else if (kind === 'file') {
      srcUrl = URL.createObjectURL(file);
    }

    const v = $('#preview');
    if (srcUrl) {
      v.src = srcUrl;
      v.muted = false;
    } else {
      v.srcObject = stream;
      v.muted = true;
    }
    v.classList.add('live');
    v.loop = kind === 'file';
    await v.play();

    if (kind === 'file') {
      stream = v.captureStream ? v.captureStream() : (v.mozCaptureStream ? v.mozCaptureStream() : null);
      if (!stream) throw new Error('Video stream capture not supported in this browser.');
    }

    if (stream && stream.getTracks()[0]) {
      stream.getTracks()[0].addEventListener('ended', stopCapture);
    }

    $('#btnStop').disabled = false;
    $('#srcStatus').textContent = 'Loading MediaPipe models...';
    if (!vision.face) await vision.init();
    const hasAudio = stream ? await audio.start(stream) : false;
    if (!hasAudio) log('No audio track was shared. Voice and lip-sync checks are off. Tick "Share audio" when picking a tab.');
    t0 = performance.now();
    visionTimer = setInterval(visionStep, 33);
    log(`Capture started (${kind === 'screen' ? 'shared window' : kind === 'cam' ? 'webcam' : 'video file'}).`);
  } catch (err) {
    $('#srcStatus').textContent = `Could not start: ${err.message}`;
    stopCapture();
  }
}

function stopCapture() {
  clearInterval(visionTimer);
  visionTimer = null;
  audio.stop();
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  const v = $('#preview');
  v.srcObject = null;
  if (v.src) {
    URL.revokeObjectURL(v.src);
    v.removeAttribute('src');
  }
  v.classList.remove('live');
  $('#btnStop').disabled = true;
}

function visionStep() {
  const t = performance.now();
  const res = vision.detect($('#preview'), t, challenge?.state === 'running');
  if (!res) return;
  if (res.face) {
    lastFaceT = t;
    videoMetrics.push(t, res.face, res.blend);
    crossModal.pushMouth(t, mouthAperture(res.face));
  }
  if (challenge?.state === 'running') stepChallenge(res, t);
}

// ---------- active challenge (part 5) ----------
function renderChallenge(trust, now) {
  const el = $('#challengeCard');
  if (!challenge && stream && trust < 0.45 && now > challengeCooldownUntil && now - callStart > 15000) {
    challenge = { steps: pickChallenge(), i: 0, state: 'suggested', hits: 0, stepStart: 0 };
  }
  if (!challenge) { el.hidden = true; return; }
  el.hidden = false;
  const prompts = challenge.steps.map((k) => CHALLENGES[k].prompt.toLowerCase());
  const ask = `"Please ${prompts[0]}, then ${prompts[1]}."`;
  const left = Math.max(0, 15 - Math.round((now - challenge.stepStart) / 1000));
  const key = `${challenge.state}|${challenge.i}|${challenge.state === 'running' ? left : ''}`;
  if (key === challenge.key) return; // avoid replacing buttons mid-click
  challenge.key = key;
  if (challenge.state === 'suggested') {
    el.innerHTML = `<h3>Trust is low</h3><div class="big">Ask the caller: ${esc(ask)}</div>` +
      `<button class="primary" id="chGo">Watch for it</button> <button id="chSkip">Dismiss</button>` +
      `<div class="steps">A live face-swap often breaks when a hand crosses the face.</div>`;
  } else if (challenge.state === 'loading') {
    el.innerHTML = '<h3>Active challenge</h3><div>Loading hand tracking...</div>';
  } else if (challenge.state === 'running') {
    el.innerHTML = `<h3>Watching the caller</h3><div class="big">Step ${challenge.i + 1} of 2: ${esc(CHALLENGES[challenge.steps[challenge.i]].prompt)}</div><div class="steps">${left}s left</div>`;
  } else {
    const ok = challenge.state === 'passed';
    el.innerHTML = `<h3>Challenge ${ok ? 'passed' : 'failed'}</h3><div>${ok ? 'The caller followed both instructions.' : 'The caller did not follow the instruction in time.'}</div>` +
      `<div class="steps">${ok ? 'This is one more point for a real person, but it does not cancel the other signals.' : ''}</div><button id="chSkip">Dismiss</button>`;
  }
  $('#chGo')?.addEventListener('click', runChallenge);
  $('#chSkip')?.addEventListener('click', () => { challenge = null; challengeCooldownUntil = performance.now() + 60000; });
}

async function runChallenge() {
  challenge.state = 'loading';
  try {
    await vision.enableHands();
  } catch (err) {
    log(`Hand tracking failed to load: ${err.message}`);
    challenge = null;
    return;
  }
  challenge.state = 'running';
  challenge.stepStart = performance.now();
  log(`Challenge started: ${challenge.steps.map((k) => CHALLENGES[k].prompt).join(' then ')}`);
}

function stepChallenge(res, t) {
  const cur = CHALLENGES[challenge.steps[challenge.i]];
  const v = $('#preview');
  const aspect = (v.videoWidth || 16) / (v.videoHeight || 9);
  const hit = res.hands.some((h) => cur.check(h, res.face, aspect));
  challenge.hits = hit ? challenge.hits + 1 : 0;
  if (challenge.hits >= 4) {
    challenge.i++;
    challenge.hits = 0;
    challenge.stepStart = t;
    if (challenge.i >= challenge.steps.length) {
      challenge.state = 'passed';
      report('CHALLENGE_FAILED', { fired: false }, t);
      log('Challenge passed.', 'pass');
    }
  } else if (t - challenge.stepStart > 15000) {
    challenge.state = 'failed';
    report('CHALLENGE_FAILED', { fired: true, severity: 0.9, detail: `Caller did not "${cur.prompt.toLowerCase()}" within 15 seconds.` }, t);
  }
}

// ---------- demo controls ----------
const SIM = {
  LIP_DRIFT: { severity: 0.75, detail: 'Simulated: audio lags the mouth by 280 ms.' },
  BLINK_CADENCE: { severity: 0.6, detail: 'Simulated: 2 blinks/min.' },
  MOTION_SMOOTHNESS: { severity: 0.55, detail: 'Simulated: facial geometry flickers between frames.' },
  VOCODER_ARTIFACT: { severity: 0.5, detail: 'Simulated: hard spectral cutoff above 8 kHz.' },
  PROSODY_FLAT: { severity: 0.55, detail: 'Simulated: pitch variation 0.6 semitones.' },
  BREATH_ABSENT: { severity: 0.6, detail: 'Simulated: no breaths before 6 phrases.' },
};

function buildSimButtons() {
  const host = $('#simButtons');
  for (const [id, s] of Object.entries(SIM)) {
    const b = document.createElement('button');
    b.textContent = SIGNALS[id].label;
    b.addEventListener('click', () => report(id, { fired: true, ...s }));
    host.append(b);
  }
}

function resetAll() {
  sigState.clear();
  compound.reset();
  videoMetrics.reset();
  crossModal.reset();
  audio.metrics.reset();
  marker = null;
  scam = null;
  challenge = null;
  challengeCooldownUntil = 0;
  t0 = performance.now();
  $('#log').innerHTML = '';
  log('New call.');
}

// ---------- overlay: drag and pop-out ----------
function makeDraggable(el, handle) {
  let sx, sy, ox, oy, down = false;
  handle.addEventListener('pointerdown', (e) => {
    if (e.target.closest('button') || el.classList.contains('pip')) return;
    down = true; sx = e.clientX; sy = e.clientY;
    const r = el.getBoundingClientRect(); ox = r.left; oy = r.top;
    handle.setPointerCapture(e.pointerId);
    handle.style.cursor = 'grabbing';
  });
  handle.addEventListener('pointermove', (e) => {
    if (!down) return;
    el.style.left = `${Math.max(0, ox + e.clientX - sx)}px`;
    el.style.top = `${Math.max(0, oy + e.clientY - sy)}px`;
    el.style.right = 'auto';
  });
  handle.addEventListener('pointerup', () => { down = false; handle.style.cursor = ''; });
}

async function popOut() {
  const ov = $('#overlay');
  if (!('documentPictureInPicture' in window)) {
    return log('Pop-out needs Chrome or Edge 116+. Drag the panel over the meeting instead.');
  }
  if (ov.classList.contains('pip')) return;
  const w = await window.documentPictureInPicture.requestWindow({ width: 400, height: 720 });
  const link = w.document.createElement('link');
  link.rel = 'stylesheet';
  link.href = new URL('styles.css', location.href).href;
  w.document.head.append(link);
  w.document.body.style.margin = '0';
  w.document.body.style.background = 'var(--bg, #0b0f17)';
  ov.classList.add('pip');
  w.document.body.append(ov);
  w.addEventListener('pagehide', () => {
    ov.classList.remove('pip');
    document.body.append(ov);
  });
}

// ---------- init ----------
async function init() {
  try {
    config = await api('/api/config');
  } catch { /* server offline: the UI still loads */ }
  $('#badgeIpqs').classList.toggle('on', config.ipqs);
  $('#badgeIpqs').textContent = config.ipqs ? 'Number lookup: IPQualityScore' : 'Number lookup: demo data';
  $('#badgeLlm').classList.toggle('on', config.llm);
  $('#badgeLlm').textContent = config.llm ? 'Scam classifier: Claude' : 'Scam classifier: offline rules';

  const sel = $('#scenario');
  for (const s of config.scenarios || []) sel.append(new Option(s.label, s.id));
  sel.addEventListener('change', () => {
    currentDemo = sel.value;
    const s = config.scenarios.find((x) => x.id === currentDemo);
    if (!s) return;
    $('#number').value = s.call.number;
    $('#claimedOrg').value = s.call.claimedOrg;
    $('#verstat').value = s.call.verstat;
    $('#originCountry').value = s.call.originCountry;
    $('#claimedLocal').checked = s.call.claimedLocal;
    $('#userLoc').value = `${s.call.user.country} / ${s.call.user.region}`;
  });
  $('#number').addEventListener('input', () => { currentDemo = ''; });

  $('#btnCheck').addEventListener('click', async () => {
    await checkNumber();
    if ($('#transcript').value.trim() && !scam) classifyNow();
  });
  $('#btnGeo').addEventListener('click', useGeolocation);
  $('#btnLatest').addEventListener('click', async () => {
    const c = await api('/api/calls/latest');
    if (!c) return ($('#metaStatus').textContent = 'No Twilio webhook call received yet.');
    currentDemo = '';
    $('#number').value = c.number;
    $('#originCountry').value = c.originCountry;
    $('#verstat').value = c.verstat;
    checkNumber();
  });
  $('#btnScreen').addEventListener('click', () => startCapture('screen'));
  $('#btnCam').addEventListener('click', () => startCapture('cam'));
  $('#btnVideoFile').addEventListener('click', () => $('#fileUpload').click());
  $('#fileUpload').addEventListener('change', (e) => {
    if (e.target.files.length) startCapture('file', e.target.files[0]);
    e.target.value = '';
  });
  $('#btnStop').addEventListener('click', stopCapture);
  $('#btnClassify').addEventListener('click', classifyNow);
  $('#btnSpeech').addEventListener('click', toggleSpeech);
  $('#btnReset').addEventListener('click', resetAll);
  $('#btnMin').addEventListener('click', () => $('#overlay').classList.toggle('min'));
  $('#btnPip').addEventListener('click', popOut);

  buildSimButtons();
  makeDraggable($('#overlay'), $('#dragHandle'));
  log('Ready. Check a number, share a meeting window, or use the demo controls.');
  setInterval(tick, 1000);
  tick();
}

init();
