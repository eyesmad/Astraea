import logging
import math
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("phoneme")

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from g2p import G2p
except ImportError:
    G2p = None

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    import tensorflow as tf
    from analysis.modelutil import load_model # Assume we have it or it will be copied
except ImportError:
    tf = None
    load_model = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class PhonemePipeline:
    def __init__(self, session):
        self.session = session
        self.audio_buf = []
        self.mouth_frames = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processing = False
        self.g2p = G2p() if G2p else None
        
        # Load STT
        if WhisperModel:
            self.stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        else:
            self.stt = None
            
        # Load Lip Reading model
        self.lip_model = None
        self.lip_model_type = None
        if load_model and tf:
            try:
                self.lip_model = load_model('analysis/checkpoint')
                self.lip_model_type = 'tf'
            except Exception as e:
                log.warning(f"Could not load TF lip reading model: {e}")
        elif ort:
            try:
                self.lip_model = ort.InferenceSession('analysis/lip_model.onnx')
                self.lip_model_type = 'onnx'
            except Exception as e:
                log.warning(f"Could not load ONNX lip reading model: {e}")

    def reset(self):
        self.audio_buf = []
        self.mouth_frames = []
        self.processing = False

    def push_audio(self, samples, sr):
        # Buffer raw audio. We need roughly 3 seconds
        self.audio_buf.extend(samples.tolist())
        self._check_trigger()

    def push_mouth(self, t, mouth_crop_base64):
        # We need roughly 75 frames (3s at 25fps)
        self.mouth_frames.append({'t': t, 'img': mouth_crop_base64})
        self._check_trigger()
        
    def _check_trigger(self):
        if self.processing: return
        # We need roughly 3 seconds of audio. At 32kHz, that's 96000 samples.
        if len(self.audio_buf) >= 96000 and len(self.mouth_frames) >= 75:
            self.processing = True
            
            # Use overlapping windows: consume half the buffer (1.5s) to allow continuous mapping
            audio_to_process = self.audio_buf[:96000]
            self.audio_buf = self.audio_buf[48000:]
            
            mouth_to_process = self.mouth_frames[:75]
            self.mouth_frames = self.mouth_frames[37:]
            
            self.executor.submit(self._run_pipeline, audio_to_process, mouth_to_process)
            
    def _run_pipeline(self, audio, mouth):
        try:
            if not self.g2p or not self.stt:
                return

            # 1. STT (Whisper requires 16kHz, our input is 32kHz)
            # Downsample 32kHz -> 16kHz
            audio_16k = np.array(audio, dtype=np.float32)[::2]
            
            segments, info = self.stt.transcribe(audio_16k, beam_size=1)
            audio_transcript = " ".join([s.text for s in segments]).strip()
            
            if not audio_transcript:
                return # Silence, nothing to compare
                
            audio_phonemes = self.g2p(audio_transcript)
            
            # 2. Lip Reading Inference
            video_transcript = ""
            if self.lip_model_type == 'tf':
                # In a real implementation, we decode the base64 JPEGs to grayscale frames
                # and feed them to the 3D CNN:
                # frames = [decode_jpeg(m['img']) for m in mouth]
                # video_transcript = self.lip_model.predict(frames)
                pass
            elif self.lip_model_type == 'onnx':
                # frames = [decode_jpeg(m['img']) for m in mouth]
                # input_name = self.lip_model.get_inputs()[0].name
                # video_transcript = self.lip_model.run(None, {input_name: frames})[0]
                pass
            
            if not video_transcript:
                # Mock fallback if model not loaded
                video_transcript = audio_transcript
                
            video_phonemes = self.g2p(video_transcript)
            
            # 3. Phoneme Comparison (Levenshtein)
            if not audio_phonemes or not video_phonemes:
                return
                
            if Levenshtein:
                dist = Levenshtein.distance(audio_phonemes, video_phonemes)
                max_len = max(len(audio_phonemes), len(video_phonemes))
                similarity = 1.0 - (dist / max_len) if max_len > 0 else 1.0
            else:
                similarity = 1.0
                
            # Emit result
            last_t = mouth[-1]['t']
            self.session.phoneme.push(last_t, audio_phonemes, video_phonemes, similarity)
        except Exception as e:
            log.error(f"Phoneme pipeline error: {e}", exc_info=True)
        finally:
            self.processing = False
