// MediaPipe Tasks Vision wrapper: Face Mesh (always) and Hands (only while an active challenge runs).
const VERSION = '0.10.14';
const BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}`;
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
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
    this.face = await this._create('FaceLandmarker', FACE_MODEL, { numFaces: 1, outputFaceBlendshapes: true });
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

  // Returns { face, blend, hands } for the current video frame, or null if the frame has not advanced.
  detect(video, tMs, withHands = false) {
    if (video.readyState < 2 || video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    const f = this.face.detectForVideo(video, tMs);
    const face = f.faceLandmarks?.[0] || null;
    let blend = null;
    const cats = f.faceBlendshapes?.[0]?.categories;
    if (cats) {
      blend = {};
      for (const c of cats) blend[c.categoryName] = c.score;
    }
    let hands = [];
    if (withHands && this.hands) hands = this.hands.detectForVideo(video, tMs + 0.01).landmarks || [];
    return { face, blend, hands };
  }
}
