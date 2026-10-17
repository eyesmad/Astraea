// MediaPipe Tasks Vision wrapper: Face Mesh (always) and Hands (only while an active challenge runs).
const VERSION = '0.10.14';
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
export const MAX_FACES = 4; // more than one caller can be on screen; metrics follow the largest
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export class Vision {
  constructor() {
    this.face = null;
    this.hands = null;
    this.lib = null;
    this.fileset = null;
    this.lastVideoTime = -1;
  }

  async init() {
    this.lib = await import(`${BASE}/vision_bundle.mjs`);
    this.fileset = await this.lib.FilesetResolver.forVisionTasks(`${BASE}/wasm`);
    this.face = await this._create('FaceLandmarker', FACE_MODEL, { numFaces: MAX_FACES, outputFaceBlendshapes: true });
  }

  async _create(kind, modelAssetPath, extra) {
    const make = (delegate) => this.lib[kind].createFromOptions(this.fileset, {
      baseOptions: { modelAssetPath, delegate },
      runningMode: 'VIDEO',
      ...extra,
    });
    try {
      return await make('GPU');
    } catch {
      return await make('CPU');
    }
  }

  async enableHands() {
    if (!this.hands) this.hands = await this._create('HandLandmarker', HAND_MODEL, { numHands: 2 });
  }

  // Returns { faces, face, blend, hands } for the current frame, or null if the frame has not advanced.
  // `faces` is every detected face, largest first; `face`/`blend` are the largest one, which the metrics follow.
  detect(video, tMs, withHands = false) {
    if (video.readyState < 2 || video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    const f = this.face.detectForVideo(video, tMs);

    const faces = (f.faceLandmarks || []).map((lm, i) => {
      const cats = f.faceBlendshapes?.[i]?.categories;
      let blend = null;
      if (cats) {
        blend = {};
        for (const c of cats) blend[c.categoryName] = c.score;
      }
      return { lm, blend, box: boundingBox(lm) };
    }).sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h);

    let hands = [];
    if (withHands && this.hands) hands = this.hands.detectForVideo(video, tMs + 0.01).landmarks || [];
    return { faces, face: faces[0]?.lm || null, blend: faces[0]?.blend || null, hands };
  }
}

// Normalised bounding box (0-1) around every landmark of one face.
export function boundingBox(lm) {
  let x0 = 1;
  let y0 = 1;
  let x1 = 0;
  let y1 = 0;
  for (const p of lm) {
    if (p.x < x0) x0 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x;
    if (p.y > y1) y1 = p.y;
  }
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
