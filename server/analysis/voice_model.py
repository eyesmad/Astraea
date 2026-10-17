"""Optional pellav2 voice-deepfake model, loaded in-process.

The model is a bonus signal, never a dependency: if torch is missing or the weights cannot be fetched, the
signal is simply absent and the heuristics carry the voice layer on their own. Loading takes minutes and
scoring takes seconds, so both happen on a worker thread and results arrive through a callback rather than
blocking the asyncio loop that serves the WebSocket.

Enable with VOICE_MODEL=1. Weights (about 1.2 GB) download from the Hugging Face Hub on first use.
"""
import importlib.util
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("voice-model")

REPO = os.environ.get("VOICE_MODEL_REPO", "Sadanie/pellav2-audio-deepfake-detector")


def enabled(env=None):
    env = os.environ if env is None else env
    return bool(env.get("VOICE_MODEL")) and env.get("VOICE_MODEL") != "0"


class VoiceModel:
    def __init__(self):
        self.ready = False
        self.failed = None
        self._model = None
        self._torch = None
        self._crop = 64000
        self._busy = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-model")
        self._pool.submit(self._load)

    @property
    def busy(self):
        return self._busy or not self.ready

    def _load(self):
        try:
            import torch
            from huggingface_hub import hf_hub_download

            spec = importlib.util.spec_from_file_location(
                "pellav2_infer", hf_hub_download(REPO, "pellav2_infer.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            model = mod.Detector()
            weights = os.environ.get("VOICE_MODEL_WEIGHTS") or hf_hub_download(REPO, "pellav2_detector.pt")
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            model.eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # leave room for the server

            self._torch, self._model, self._crop = torch, model, mod.CROP
            self.ready = True
            log.info("Voice model ready.")
        except Exception as err:
            self.failed = str(err)
            log.warning("Voice model unavailable (%s); using voice heuristics only.", err)

    def _score(self, pcm):
        import numpy as np

        torch = self._torch
        x = np.asarray(pcm, dtype=np.float32)
        if len(x) >= self._crop:
            off = (len(x) - self._crop) // 2
            x = x[off : off + self._crop]
        else:
            x = np.pad(x, (0, self._crop - len(x)))
        x = (x - x.mean()) / (x.std() + 1e-7)
        with torch.no_grad():
            return float(torch.sigmoid(self._model(torch.from_numpy(x)[None])).item())

    def submit(self, pcm, callback):
        """Scores a window on the worker thread; callback(score) runs there too. Drops if already busy."""
        if self.busy:
            return False
        self._busy = True

        def run():
            score = None
            try:
                score = self._score(pcm)
            except Exception as err:
                log.warning("Scoring failed: %s", err)
            finally:
                self._busy = False
            try:
                callback(score)
            except Exception:
                log.exception("Voice model callback failed")

        self._pool.submit(run)
        return True


_shared = None
_lock = threading.Lock()


def voice_model():
    """One model for the whole server: loading the weights takes far longer than scoring a window."""
    global _shared
    if not enabled():
        return None
    with _lock:
        if _shared is None:
            log.info("Voice model enabled, loading weights (first run downloads about 1.2 GB)...")
            _shared = VoiceModel()
    return _shared
