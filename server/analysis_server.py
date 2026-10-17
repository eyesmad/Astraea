"""Astraea analysis server.

The browser streams 32 kHz PCM and compact face frames to /ws/analyze, which the Node server proxies here;
this process runs the voice, face and cross-modal heuristics and streams signals back once a second.

Run it alongside the Node server with `npm start`, which starts both halves.
"""
import asyncio
import json
import logging
import struct

import numpy as np
import websockets

from analysis import SAMPLE_RATES
from analysis.recording import RecordingWriter, recording_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("analysis")

PORT = 3002
MAX_CHUNK_SAMPLES = 48000
TICK_SECONDS = 1.0

import logging
import math
import time

from analysis_metrics import (
    spectrum_db, estimate_f0, brickwall_drop, spectral_flatness, rms,
    unpack_face, mouth_aperture, AudioMetrics, VideoMetrics, CrossModal,
    CompoundDetector, PhonemeSync
)
from analysis.phoneme_pipeline import PhonemePipeline

logging.basicConfig(level=logging.INFO)

class AnalysisSession:
    def __init__(self, sample_rate=32000, fft_size=2048, hop_ms=20):
        self.sr = sample_rate
        self.fft_size = fft_size
        self.hop = int((sample_rate * hop_ms) / 1000)
        self.hop_ms = (self.hop / sample_rate) * 1000
        self.audio = AudioMetrics()
        self.video = VideoMetrics()
        self.cross = CrossModal()
        self.compound = CompoundDetector()
        self.phoneme = PhonemeSync()
        self.phoneme_pipeline = PhonemePipeline(self)
        self.reset()

    def reset(self):
        self.audio.reset()
        self.video.reset()
        self.cross.reset()
        self.compound.reset()
        self.phoneme.reset()
        self.phoneme_pipeline.reset()
        self.ring = np.zeros(self.fft_size, dtype=np.float32)
        self.total = 0
        self.filled = 0
        self.since_hop = 0
        self.offset = None
        self.now = 0
        self.tick = 0
        self.last_frame_t = 0
        self.f0 = {'f0': 0.0, 'clarity': 0.0}
        self.drop = {'dropDb': 0.0, 'atHz': 0.0}
        self.faces = 0
        self.audio_buf = np.zeros(0, dtype=np.float32)

    def push_audio(self, samples, end_t):
        end_idx = self.total + len(samples)
        off = end_t - (end_idx / self.sr) * 1000.0
        if self.offset is None:
            self.offset = off
        else:
            self.offset = min(off, self.offset + 0.02)
            
        self.phoneme_pipeline.push_audio(samples, self.sr)
        self.audio_buf = np.concatenate((self.audio_buf, samples))
        
        while self.since_hop + len(samples) >= self.hop:
            consume = self.hop - self.since_hop
            self.total += consume
            samples = samples[consume:]
            self.since_hop = 0
            
            frame_end_idx = len(self.audio_buf) - len(samples)
            if frame_end_idx >= self.fft_size:
                frame = self.audio_buf[frame_end_idx - self.fft_size : frame_end_idx]
                self._frame(self.offset + (self.total / self.sr) * 1000.0, frame)
        
        self.total += len(samples)
        self.since_hop += len(samples)
        
        # Keep enough history for the next window plus the chunk that will be appended to it.
        if len(self.audio_buf) > self.fft_size * 3:
            self.audio_buf = self.audio_buf[-(self.fft_size * 2):]

    def _frame(self, t, frame):
        n = self.fft_size
        level = rms(frame)
        self.cross.push_audio(t, level)
        self.now = max(self.now, t)

        freq = spectrum_db(frame)
        hz = self.sr / n
        flat = spectral_flatness(freq, math.floor(300 / hz), math.floor(4000 / hz))
        
        if level >= 0.02 and self.tick % 2 == 0:
            self.f0 = estimate_f0(frame, self.sr)
            self.drop = brickwall_drop(freq, self.sr, n)
        self.tick += 1
        
        dt = t - self.last_frame_t if self.last_frame_t else self.hop_ms
        self.last_frame_t = t
        self.audio.push({
            't': t, 'dt': dt, 'rms': level, 'flatness': flat,
            'f0': self.f0['f0'], 'clarity': self.f0['clarity'], 'dropDb': self.drop['dropDb']
        })

    def push_face(self, t, face_data, mouth_crop=None):
        unpacked = unpack_face(face_data)
        if not unpacked:
            return False
        
        self.video.push(t, unpacked['lm'], unpacked['blend'])
        if unpacked['lm']:
            self.cross.push_mouth(t, mouth_aperture(unpacked['lm']))
        
        if mouth_crop:
            self.phoneme_pipeline.push_mouth(t, mouth_crop)

        self.now = max(self.now, t)
        return True

    def evaluate(self):
        results = self.video.evaluate() + self.audio.evaluate() + self.cross.evaluate(self.now) + self.phoneme.evaluate()
        for r in results:
            self.compound.report(r['id'], r.get('fired', False), r.get('severity', 0), self.now)
        
        # In the original JS the compound detector evaluated directly.
        # We don't necessarily need to return compound signals unless we want to, 
        # but the JS client runs its own compound detector locally based on received signals.
        return results

    def probes(self):
        return {
            "video": self.video.probe(),
            "voice": self.audio.probe(),
            "crossmodal": self.cross.probe(self.now),
            "phoneme": self.phoneme.probe()
        }
    
    @property
    def frames(self):
        return self.video.frames

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

async def handler(websocket):
    session = AnalysisSession()
    recorder = None
    log.info("Client connected")

    async def send_updates():
        try:
            while True:
                await asyncio.sleep(TICK_SECONDS)
                # evaluate() is CPU-bound; a thread keeps the socket responsive while it runs.
                results, probes = await asyncio.to_thread(lambda: (session.evaluate(), session.probes()))
                # NumpyEncoder exists for exactly this: numpy bools and floats reach here from the
                # metrics and json.dumps refuses them without it.
                await websocket.send(json.dumps({
                    "type": "signals",
                    "results": results,
                    "probes": probes,
                    "frames": session.frames,
                    "faces": session.faces,
                }, cls=NumpyEncoder))
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:
            logging.error(f"Error in send_updates task: {e}", exc_info=True)

    task = asyncio.create_task(send_updates())

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                if len(message) <= 8:
                    continue
                count = (len(message) - 8) // 2
                if count > MAX_CHUNK_SAMPLES:
                    continue
                end_t = struct.unpack("<d", message[:8])[0]
                samples = np.frombuffer(message[8 : 8 + count * 2], dtype="<i2").astype(np.float32) / 32768.0
                session.push_audio(samples, end_t)
                if recorder:
                    recorder.audio(message)
                continue

            msg = json.loads(message)
            kind = msg.get("type")

            if kind == "config":
                rate = msg.get("sampleRate", 32000)
                if rate not in SAMPLE_RATES:
                    continue
                session = AnalysisSession(sample_rate=rate)
                if recorder:
                    recorder.close()
                    recorder = None
                label = msg.get("record")
                if label:
                    note = msg.get("note")
                    path = recording_path(label)
                    recorder = RecordingWriter(path, rate, label, note[:500] if isinstance(note, str) else "")
                    log.info("Recording capture to %s", path)
                await websocket.send(json.dumps({"type": "ready", "recording": bool(recorder)}))

            elif kind == "faces":
                # Several faces may be visible; metrics track the primary one the client nominates.
                faces = msg.get("faces") or []
                session.faces = len(faces)
                primary = next((f for f in faces if f.get("primary")), faces[0] if faces else None)
                if primary and session.push_face(primary.get("t", msg.get("t", 0)), primary.get("p")):
                    if recorder:
                        recorder.face(primary.get("t", msg.get("t", 0)), primary.get("p"))

            elif kind == "face":  # single-face form, kept so older clients and recordings still work
                t = msg.get("t", 0)
                session.faces = 1
                if session.push_face(t, msg.get("p")) and recorder:
                    recorder.face(t, msg.get("p"))

            elif kind == "reset":
                session.reset()

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logging.error(f"Error in websocket loop: {e}", exc_info=True)
    finally:
        task.cancel()
        if recorder:
            recorder.close()
        log.info("Client disconnected")


async def main():
    log.info("Analysis server listening on 127.0.0.1:%d", PORT)
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=2 ** 20):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
