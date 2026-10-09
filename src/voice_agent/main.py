"""Standalone terminal runner with live T0-T6 latency telemetry.

Usage:
  python -m voice_agent.main            # needs env keys, mic + speakers
  python -m voice_agent.main --debug    # + mic/STT health logs (use when turns go missing)
  python -m voice_agent.main --list-devices
  python -m voice_agent.main --no-audio  # pipeline smoke test (no hardware)
  python -m voice_agent.main --check-mic # 5 s mic self-test (no cloud keys needed)

Latency model (budgets from the blueprint):
  T_speech_end --80ms--> T_endpoint --140ms--> T_first_token --120ms--> T_first_audio
  Total TTFA target: 350-480 ms (assert < 500 ms).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
import time

sys.path.insert(0, "src")

from voice_agent.audio.device import AudioDeviceManager
from voice_agent.config import AgentConfig
from voice_agent.pipeline.session import VoiceSessionCoordinator
from voice_agent.providers.llm_groq import GroqStreamingLLM
from voice_agent.providers.stt_deepgram import DeepgramPersistentSTT
from voice_agent.providers.tts_cartesia import create_tts_provider
from voice_agent.vad.onnx_vad import SileroVADONNX

try:
    from colorama import Fore, Style
    from colorama import init as color_init
    color_init()
    GREEN, CYAN, YELLOW, DIM = Fore.GREEN, Fore.CYAN, Fore.YELLOW, Style.DIM
    RESET = Style.RESET_ALL
except Exception:
    GREEN = CYAN = YELLOW = DIM = RESET = ""


def print_telemetry(t: dict) -> None:
    def ms(a: str, b: str) -> str:
        if a in t and b in t:
            return f"{(t[b]-t[a])*1000:.0f}ms"
        return "n/a"
    end, aud = t.get("T_turn_end"), t.get("T_first_audio")
    total = f"{(aud-end)*1000:.0f}ms" if end and aud else "n/a"
    flag = f"{GREEN}OK{RESET}" if (end and aud and (aud - end) < 0.5) else f"{YELLOW}OVER{RESET}"
    print(f"{DIM}[telemetry] TTFT {ms('T_turn_end','T_first_token')} | "
          f"CHUNK {ms('T_first_token','T_tts_start')} | "
          f"TTS {ms('T_tts_start','T_first_audio')} | "
          f"LLM {ms('T_first_token','T_llm_done')} | TTFA {total} [{flag}]{RESET}")


async def run_no_audio(cfg: AgentConfig) -> int:
    print(f"{CYAN}voice-agent-v2 smoke test (no audio hardware).{RESET}")
    from voice_agent.pipeline.chunker import AdaptiveClauseChunker
    c = AdaptiveClauseChunker()
    toks = ["Sure,", "I", "can", "help", "with", "that", "today."]
    for i, w in enumerate(toks):
        out = c.push(w + (" " if i < len(toks)-1 else ""))
        for p in out:
            print(f"  chunk -> {p!r}")
    for p in c.flush():
        print(f"  chunk -> {p!r}")
    print(f"{GREEN}chunker OK{RESET} | VAD model: {cfg.vad_model_path} | "
          f"LLM: {cfg.groq_model} | TTS: {cfg.tts_provider}")
    print("Set DEEPGRAM_API_KEY/GROQ_API_KEY + TTS key in .env for a live run.")
    return 0


async def run_check_mic(seconds: float = 5.0) -> int:
    """Mic self-test: no cloud keys needed. Proves capture + VAD locally."""
    import numpy as _np
    print(f"{CYAN}Mic self-test: speak for {seconds:.0f}s — bars should jump with your voice.{RESET}")
    audio = AudioDeviceManager()
    loop = asyncio.get_running_loop()
    try:
        audio.start(loop)
    except Exception as e:
        print(f"Audio start failed: {e}")
        return 1
    try:
        from voice_agent.vad.onnx_vad import SileroVADONNX
        try:
            vad = SileroVADONNX()
        except Exception as e:
            print(f"{YELLOW}VAD unavailable ({e}); showing levels only.{RESET}")
            vad = None
    except Exception:
        vad = None
    deadline = time.monotonic() + seconds
    peak = 0.0
    frames = 0
    try:
        while time.monotonic() < deadline:
            try:
                chunk = await asyncio.wait_for(audio.input_queue.get(), timeout=1.0)
            except TimeoutError:
                print(f"{YELLOW}no mic frames for 1s — wrong device or muted mic. Run --list-devices.{RESET}")
                return 3
            frames += 1
            s = _np.frombuffer(chunk, dtype=_np.int16).astype(_np.float32)
            rms = float(_np.sqrt(_np.mean(s * s)) / 32768.0)
            peak = max(peak, rms)
            bar = "#" * min(40, int(rms * 400))
            speech = ""
            if vad is not None:
                with contextlib.suppress(Exception):
                    speech = " SPEECH" if vad.is_speech(chunk) else ""
            print(f"\rrms={rms:.4f} [{bar:<40}] peak={peak:.4f}{speech}   ", end="", flush=True)
    finally:
        print()
        audio.stop()
    print(f"{GREEN}mic OK{RESET}: {frames} frames, peak rms={peak:.4f} "
          f"({'silent — mic too quiet/muted?' if peak < 0.005 else 'levels look healthy'})")
    return 0


async def run_live(cfg: AgentConfig, debug: bool = False) -> int:
    audio = AudioDeviceManager(sample_rate=cfg.sample_rate, frame_size=cfg.frame_size)
    if cfg.aec_enabled:
        from voice_agent.audio.aec import create_canceller
        from voice_agent.audio.speaker import SpeakerClassifier
        audio.enable_aec(create_canceller(cfg.aec_backend,
                                          stream_delay_ms=cfg.aec_delay_ms),
                         SpeakerClassifier())
        print(f"{DIM} AEC on ({cfg.aec_backend}): mic cleaned, labels in telemetry.{RESET}")
    vad = SileroVADONNX(model_path=cfg.vad_model_path, threshold=cfg.vad_threshold)
    llm = GroqStreamingLLM(api_key=cfg.groq_api_key, model=cfg.groq_model,
                           system_prompt=cfg.system_prompt,
                           temperature=cfg.llm_temperature, max_tokens=cfg.llm_max_tokens)
    tts = create_tts_provider(cfg)

    # Debug sink is the logger itself: single lines via logging (the old
    # print wrapper doubled every diagnostic line).
    _dbg = logging.getLogger("voice_agent.session").debug if debug else None

    async def announce(text: str) -> None:
        print(f"{GREEN}assistant:{RESET} {text}")

    # Session-bound callbacks defined before either object exists; they
    # resolve `session` late (at call time, never at definition).
    async def _on_turn(text: str, confidence: float | None = None) -> None:
        await session.on_turn_complete(text, confidence)

    def _on_started() -> None:
        session.note_speech_started()

    stt = DeepgramPersistentSTT(api_key=cfg.deepgram_api_key,
                                 on_turn_complete=_on_turn,
                                 model=cfg.deepgram_model,
                                 endpointing_ms=cfg.deepgram_endpointing_ms,
                                 log_debug=_dbg)
    # interim transcripts: liveness print + text-confirmed barge-in
    async def _interim(text: str) -> None:
        print(f"\r{DIM}... {text[:80]}{RESET}", end="", flush=True)
        await session.on_interim_text(text)
    stt.on_interim = _interim  # type: ignore
    stt.on_speech_started = _on_started  # type: ignore

    session = VoiceSessionCoordinator(audio, vad, stt, llm, tts,
                                     barge_in_frames=cfg.barge_in_frames,
                                     barge_in_frames_speaking=cfg.barge_in_frames_speaking,
                                     barge_in_mic_floor=cfg.barge_in_mic_floor,
                                     barge_in_rise_ratio=cfg.barge_in_rise_ratio,
                                     barge_in_fast_ratio=cfg.barge_in_fast_ratio,
                                     barge_in_blank_s=cfg.barge_in_blank_s,
                                     echo_recency_s=cfg.echo_recency_s,
                                     barge_in_mode=cfg.barge_in_mode,
                                     barge_in_min_novel_words=cfg.barge_in_min_novel_words,
                                     barge_in_min_novel_ratio=cfg.barge_in_min_novel_ratio,
                                     validator_min_confidence=cfg.validator_min_confidence,
                                     validator_barge_window_s=cfg.validator_barge_window_s,
                                     validator_multi_confidence=cfg.validator_multi_confidence,
                                     validator_soup_ratio=cfg.validator_soup_ratio,
                                     log_debug=_dbg,
                                     on_assistant_text=announce,
                                     on_telemetry=print_telemetry)

    print(f"{CYAN}Listening — speak anytime (barge-in enabled). Ctrl+C to quit.{RESET}")
    print(f"{DIM} LLM={cfg.groq_model} STT={cfg.deepgram_model} "
          f"TTS={cfg.tts_provider} VAD_thr={cfg.vad_threshold} "
          f"AEC={'on' if cfg.aec_enabled else 'off'}{RESET}")
    try:
        await session.start()
        while True:
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        # Python 3.11+: first Ctrl+C cancels the main task (KeyboardInterrupt
        # is only raised on a second one), so shut down here.
        print("\nShutting down...")
    finally:
        try:
            await asyncio.wait_for(session.stop(), 3.0)
        except Exception as e:
            logging.getLogger("voice_agent").warning("stop failed: %r", e)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="voice-agent-v2 terminal runner")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="mic/STT health + drop-reason logs (use when turns go missing)")
    ap.add_argument("--check-mic", action="store_true",
                    help="5 s mic self-test, no cloud keys needed")
    ap.add_argument("--mic-seconds", type=float, default=5.0)
    args = ap.parse_args()
    if args.list_devices:
        print(AudioDeviceManager.list_devices())
        return 0
    if args.check_mic:
        return asyncio.run(run_check_mic(args.mic_seconds))
    try:
        cfg = AgentConfig()  # type: ignore[call-arg]  # env provides required keys
    except Exception as e:
        print(f"Config error (.env missing keys?): {e}")
        print("Copy .env.example -> .env and fill DEEPGRAM_API_KEY, GROQ_API_KEY, and a TTS key.")
        return 2
    if args.no_audio:
        return asyncio.run(run_no_audio(cfg))
    from voice_agent.observability.logging import configure_logging
    configure_logging(debug=args.debug)
    try:
        return asyncio.run(run_live(cfg, debug=args.debug))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
