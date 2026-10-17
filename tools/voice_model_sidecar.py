"""Scores 4-second audio windows with the pellav2 deepfake detector.

Astraea runs this as a child process and talks to it over stdin/stdout, so the Node server never loads a
Python runtime itself and the model is optional: if this script cannot start, the voice-model signal is
simply absent and the heuristics carry on.

Protocol, both directions length-prefixed so partial reads cannot desync the stream:
  in   4-byte little-endian sample count, then that many little-endian float32 samples at 16 kHz mono
  out  one JSON line per request: {"score": 0.0-1.0} where higher means more likely synthetic,
       or {"error": "..."} if that window could not be scored

Requires: torch, transformers, soundfile, numpy, huggingface_hub. Weights (about 1.2 GB) download from the
Hugging Face Hub on first use and are cached.
"""
import importlib.util
import json
import os
import struct
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SR = 16000


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_model():
    import torch
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("VOICE_MODEL_REPO", "Sadanie/pellav2-audio-deepfake-detector")
    spec = importlib.util.spec_from_file_location("pellav2_infer", hf_hub_download(repo, "pellav2_infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.Detector()
    weights = os.environ.get("VOICE_MODEL_WEIGHTS") or hf_hub_download(repo, "pellav2_detector.pt")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # leave room for the Node server
    return torch, model, mod.CROP


def read_exactly(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    import numpy as np

    torch, model, crop = load_model()
    log("ready")
    print(json.dumps({"ready": True}), flush=True)

    stdin = sys.stdin.buffer
    while True:
        header = read_exactly(stdin, 4)
        if header is None:
            return
        (count,) = struct.unpack("<I", header)
        if count == 0 or count > 10 * SR:
            print(json.dumps({"error": f"bad sample count {count}"}), flush=True)
            return
        payload = read_exactly(stdin, count * 4)
        if payload is None:
            return
        try:
            x = np.frombuffer(payload, dtype="<f4").astype(np.float32)
            if len(x) >= crop:
                off = (len(x) - crop) // 2
                x = x[off:off + crop]
            else:
                x = np.pad(x, (0, crop - len(x)))
            x = (x - x.mean()) / (x.std() + 1e-7)
            with torch.no_grad():
                score = torch.sigmoid(model(torch.from_numpy(x)[None])).item()
            print(json.dumps({"score": score}), flush=True)
        except Exception as err:  # keep serving: one bad window must not kill the sidecar
            print(json.dumps({"error": str(err)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        log(f"fatal: {err}")
        sys.exit(1)
