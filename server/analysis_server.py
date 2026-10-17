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

from analysis import SAMPLE_RATES, AnalysisSession
from analysis.recording import RecordingWriter, recording_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("analysis")

PORT = 3002
MAX_CHUNK_SAMPLES = 48000
TICK_SECONDS = 1.0


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
                await websocket.send(json.dumps({
                    "type": "signals",
                    "results": results,
                    "probes": probes,
                    "frames": session.frames,
                    "faces": session.faces,
                }))
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass

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
                await asyncio.to_thread(session.push_audio, samples, end_t)
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
