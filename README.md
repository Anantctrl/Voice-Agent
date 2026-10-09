# voice-agent-v2 — Sub-500ms Real-Time Voice AI Agent

Full-duplex engine: persistent Deepgram WS + Silero VAD (ONNX) + Groq streaming LLM +
2-tier clause chunker + streaming TTS (ElevenLabs v4 / Cartesia), orchestrated with
generation-token cancellation, layered barge-in (energy + transcript + suspicion
window), optional WebRTC AEC3 echo cancellation with speaker-label telemetry,
and transcript validation before every LLM call.

Package layout: `src/voice_agent/` (`audio/`, `vad/`, `providers/`, `pipeline/`,
`observability/`); `voice_agent_v2` remains as a backwards-compatible alias.

## Quickstart

```powershell
cd D:\v
copy .env.example .env   # fill DEEPGRAM_API_KEY, GROQ_API_KEY, ELEVENLABS_API_KEY
pip install -e .         # or: pip install sounddevice numpy onnxruntime websockets httpx groq pydantic-settings colorama
python -m voice_agent.main --no-audio   # smoke test, no hardware
python -m voice_agent.main --check-mic  # 5 s mic self-test, no cloud keys needed
python -m voice_agent.main --debug      # live run with mic/STT health logs
pytest -q
```

VAD model (`silero_vad.onnx`) auto-downloads on first run.
Set `AEC_ENABLED=1` in `.env` for the experimental echo canceller.
