"""Silero VAD v5 via ONNX Runtime (Component 3) — GIL-free inference.

- Input tensor: [1, 512] float32 in [-1, 1] (32 ms @ 16 kHz).
- Recurrent state ``h,c`` shape is model-dependent: v4 uses [2,1,64],
  v5 uses [2,1,128]. We discover it from the model graph at load time
  instead of hard-coding, so either checkpoint works.
- Runs in C++ ORT threads (no GIL) at <5 ms/frame on CPU.
- Auto-downloads ``silero_vad.onnx`` (v5) on first use if missing.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise RuntimeError("onnxruntime is required: pip install onnxruntime") from e

MODEL_URLS = [
    "https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx",
    "https://huggingface.co/runanywhere/silero-vad-v5/resolve/main/silero_vad.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
]
MODEL_URL_V5 = MODEL_URLS[0]
MODEL_URL_MASTER = MODEL_URLS[1]


def ensure_model(model_path: str) -> str:
    p = Path(model_path)
    if p.exists():
        return str(p)
    # Tryrepo-local fallback locations.
    for alt in (Path.cwd() / p.name, Path(__file__).resolve().parent.parent.parent.parent / p.name):
        if alt.exists():
            return str(alt)
    last_err: Exception | None = None
    for url in MODEL_URLS:
        try:
            print(f"[vad] downloading silero_vad.onnx from {url} ...")
            urllib.request.urlretrieve(url, str(p))
            print(f"[vad] saved to {p}")
            return str(p)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not download silero_vad.onnx to {p}: {last_err}")


class SileroVADONNX:
    def __init__(self, model_path: str = "silero_vad.onnx", threshold: float = 0.5,
                 sample_rate: int = 16_000):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        resolved = ensure_model(model_path)
        self.session = ort.InferenceSession(resolved, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.sample_rate = sample_rate

        # Discover input/output names + state shape from the graph.
        self._input_names = [i.name for i in self.session.get_inputs()]
        # Conventional names: "input", "state", "sr".
        self._in_audio = self._input_names[0]
        self._in_state: str | None = None
        self._in_sr: str | None = None
        for n in self._input_names[1:]:
            ln = n.lower()
            if "state" in ln or ln in ("h", "state_h"):
                self._in_state = n
            elif ln in ("sr", "sample_rate", "sample rate"):
                self._in_sr = n
        if self._in_state is None and len(self._input_names) > 1:
            self._in_state = self._input_names[1]
        if self._in_sr is None and len(self._input_names) > 2:
            self._in_sr = self._input_names[2]

        state_shape: list[int] | None = None
        if self._in_state is not None:
            for meta in self.session.get_inputs():
                if meta.name == self._in_state:
                    # Normalize symbolic dims (None/str) -> batch 1.
                    state_shape = [int(d) if isinstance(d, int) and d > 0 else 1 for d in meta.shape]  # type: ignore
                    break
        if not state_shape:
            state_shape = [2, 1, 64]
        # Fix batch dim to 1.
        if len(state_shape) == 3:
            state_shape[1] = 1
        self._state_shape = tuple(state_shape)
        self.reset_states()

    def reset_states(self) -> None:
        self._state = np.zeros(self._state_shape, dtype=np.float32)

    def speech_prob(self, pcm16_chunk: bytes) -> float:
        audio = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.shape[0] < 512:
            audio = np.pad(audio, (0, 512 - audio.shape[0]))
        else:
            audio = audio[:512]
        audio = np.expand_dims(audio, axis=0)  # [1, 512]
        feed = {self._in_audio: audio}
        if self._in_state is not None:
            feed[self._in_state] = self._state
        if self._in_sr is not None:
            feed[self._in_sr] = np.array(self.sample_rate, dtype=np.int64)
        out, *rest = self.session.run(None, feed)
        if rest:
            self._state = rest[0].astype(np.float32, copy=False)
        prob = float(np.asarray(out).reshape(-1)[0])
        return prob

    def is_speech(self, pcm16_chunk: bytes) -> bool:
        return self.speech_prob(pcm16_chunk) >= self.threshold
