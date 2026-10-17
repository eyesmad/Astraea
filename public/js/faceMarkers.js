// Draws a marker over every face in the shared video, and keeps a stable identity for each one so a label
// does not jump between people when the detector reorders its output.

const MATCH_DISTANCE = 0.18; // normalised centre distance that still counts as the same face
const FORGET_MS = 1500; // a face that disappears for longer gets a new identity

export class FaceTracker {
  constructor() {
    this.reset();
  }

  reset() {
    this.tracks = new Map(); // id -> { id, box, t, firstT }
    this.nextId = 1;
  }

  // faces: [{ lm, blend, box }] largest first. Returns them with { id, primary } attached.
  update(faces, t) {
    for (const [id, tr] of this.tracks) if (t - tr.t > FORGET_MS) this.tracks.delete(id);

    const taken = new Set();
    return faces.map((f, i) => {
      const cx = f.box.x + f.box.w / 2;
      const cy = f.box.y + f.box.h / 2;
      let best = null;
      let bestD = MATCH_DISTANCE;
      for (const tr of this.tracks.values()) {
        if (taken.has(tr.id)) continue;
        const d = Math.hypot(cx - (tr.box.x + tr.box.w / 2), cy - (tr.box.y + tr.box.h / 2));
        if (d < bestD) {
          bestD = d;
          best = tr;
        }
      }
      const id = best ? best.id : this.nextId++;
      taken.add(id);
      this.tracks.set(id, { id, box: f.box, t, firstT: best?.firstT ?? t });
      return { ...f, id, primary: i === 0 };
    });
  }
}

// Scales the video's displayed area inside its element: the preview uses object-fit: contain, so the picture
// is letterboxed and normalised landmark coordinates must be mapped onto the drawn rectangle, not the canvas.
function contentRect(video, canvas) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const scale = Math.min(canvas.width / vw, canvas.height / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}

export function drawMarkers(canvas, video, faces, { label, tone } = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const want = { w: Math.round(canvas.clientWidth * dpr), h: Math.round(canvas.clientHeight * dpr) };
  if (canvas.width !== want.w || canvas.height !== want.h) {
    canvas.width = want.w;
    canvas.height = want.h;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!faces?.length || !video.videoWidth) return;

  const r = contentRect(video, canvas);
  const colours = { ok: '#3ecf8e', warn: '#f5b942', bad: '#ff5d6c', idle: '#8d9bb3' };
  const font = Math.max(11, Math.round(12 * dpr));

  for (const f of faces) {
    const pad = 0.02;
    const x = r.x + (f.box.x - pad) * r.w;
    const y = r.y + (f.box.y - pad) * r.h;
    const w = (f.box.w + pad * 2) * r.w;
    const h = (f.box.h + pad * 2) * r.h;
    const colour = f.primary ? colours[tone] || colours.idle : colours.idle;

    ctx.lineWidth = Math.max(1.5, (f.primary ? 2.5 : 1.5) * dpr);
    ctx.strokeStyle = colour;
    ctx.setLineDash(f.primary ? [] : [6 * dpr, 4 * dpr]);
    // Corner brackets rather than a full box: less of the face is covered.
    const c = Math.min(w, h) * 0.22;
    ctx.beginPath();
    for (const [sx, sy, dx, dy] of [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]]) {
      ctx.moveTo(sx + dx * c, sy);
      ctx.lineTo(sx, sy);
      ctx.lineTo(sx, sy + dy * c);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    const text = f.primary ? `Face ${f.id} · ${label || 'analysing'}` : `Face ${f.id} · not analysed`;
    ctx.font = `600 ${font}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width;
    const th = font + 8 * dpr;
    const ty = y - th - 4 * dpr < 0 ? y + h + 4 * dpr : y - th - 4 * dpr;
    ctx.fillStyle = 'rgba(11, 15, 23, .82)';
    ctx.fillRect(x, ty, tw + 12 * dpr, th);
    ctx.fillStyle = colour;
    ctx.fillRect(x, ty, 3 * dpr, th);
    ctx.fillStyle = '#e8edf6';
    ctx.fillText(text, x + 8 * dpr, ty + th - 7 * dpr);
  }
}
