// Part 5: active-challenge gesture checks over MediaPipe Hands (21 landmarks) and Face Mesh landmarks.
// Image y grows downward, so "up" means a smaller y.
import { dist } from './stats.js';

const TIPS = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const PIPS = { index: 6, middle: 10, ring: 14, pinky: 18 };

const extended = (h, f) => dist(h[TIPS[f]], h[0]) > dist(h[PIPS[f]], h[0]) * 1.1;

export function isThumbsUp(h) {
  if (!h || h.length < 21) return false;
  const curled = ['index', 'middle', 'ring', 'pinky'].every((f) => !extended(h, f));
  const thumbUp = h[4].y < h[3].y && h[3].y < h[2].y && h[4].y < h[0].y - 0.04;
  return curled && thumbUp;
}

export function isPeace(h) {
  if (!h || h.length < 21) return false;
  return extended(h, 'index') && extended(h, 'middle') && !extended(h, 'ring') && !extended(h, 'pinky');
}

export function isOpenPalm(h) {
  if (!h || h.length < 21) return false;
  return ['index', 'middle', 'ring', 'pinky'].every((f) => extended(h, f));
}

// Index fingertip within a fraction of face height of the nose tip. `aspect` is video width / height.
export function isTouchingNose(h, face, aspect = 16 / 9) {
  if (!h || h.length < 21 || !face) return false;
  const nose = face[1];
  const faceH = Math.hypot((face[10].x - face[152].x) * aspect, face[10].y - face[152].y) || 1;
  const d = Math.hypot((h[8].x - nose.x) * aspect, h[8].y - nose.y);
  return d < faceH * 0.18;
}

export const CHALLENGES = {
  touch_nose: { prompt: 'Touch your nose with one finger', check: (h, face, aspect) => isTouchingNose(h, face, aspect) },
  thumbs_up: { prompt: 'Give a thumbs up', check: (h) => isThumbsUp(h) },
  peace: { prompt: 'Hold up two fingers', check: (h) => isPeace(h) },
  open_palm: { prompt: 'Show an open palm to the camera', check: (h) => isOpenPalm(h) },
};

export function pickChallenge(rand = Math.random) {
  const keys = Object.keys(CHALLENGES);
  const a = keys.splice(Math.floor(rand() * keys.length), 1)[0];
  const b = keys[Math.floor(rand() * keys.length)];
  return [a, b];
}
