"""Writes the stream the browser sends to disk, so a real call can be replayed through the pipeline.

Same newline-delimited format as server/recording.js: a header line, then audio and face records in arrival
order, audio kept as the original binary chunk (base64) so a replay parses exactly what the server parsed.
"""
import base64
import json
import os
import re
from datetime import datetime, timezone

RECORDING_VERSION = 1
RECORDING_DIR = os.environ.get("RECORDING_DIR", "recordings")


def safe_label(label):
    """Labels become part of a filename and arrive from the browser, so allow only a safe slug."""
    s = re.sub(r"[^a-z0-9_-]", "", str(label or "").lower())[:40]
    return s or "capture"


def recording_path(label, directory=RECORDING_DIR):
    stamp = datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
    return os.path.join(directory, f"{safe_label(label)}-{stamp}.ndjson")


class RecordingWriter:
    def __init__(self, path, sample_rate, label, note=""):
        self.path = path
        self.audio_chunks = 0
        self.faces = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._write({
            "type": "header", "version": RECORDING_VERSION,
            "startedAt": datetime.now(timezone.utc).isoformat(),
            "sampleRate": sample_rate, "label": label, "note": note,
        })

    def _write(self, obj):
        self._fh.write(json.dumps(obj) + "\n")

    def audio(self, raw):
        self.audio_chunks += 1
        self._write({"type": "audio", "b64": base64.b64encode(raw).decode("ascii")})

    def face(self, t, p):
        self.faces += 1
        self._write({"type": "face", "t": t, "p": p})

    def close(self):
        if not self._fh.closed:
            self._fh.close()
