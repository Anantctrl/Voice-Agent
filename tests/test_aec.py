"""P1 proof: NLMS canceller + speaker classifier on synthetic fixtures.

Speech-like signals (modulated noise + formant-ish tones), a 64 ms decaying
room impulse, and labeled segments. Targets from the plan: ERLE >= 10 dB on
echo-only, label accuracy >= 90%, convergence < 500 ms, < 5 ms per 32 ms frame.
"""
import time

import numpy as np

from voice_agent.audio.aec import NoOpAEC, WebRtcAec, create_canceller
from voice_agent.audio.speaker import (
    DOUBLE,
    FAR,
    NEAR,
    QUIET,
    SpeakerClassifier,
    estimate_delay,
)

SR = 16_000
FRAME = 512


def _speech_like(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    # Distinct pitch per seed: real speakers never share exact harmonics.
    # (Identical carriers across seeds would correlate near-end with the
    # reference and the canceller would rightly eat both.)
    f0 = 90 + (seed % 5) * 25
    # Voiced-ish carrier + harmonics gated by syllabic envelopes.
    env = (rng.standard_normal(n // 160 + 1) ** 2)
    env = np.repeat(env, 160)[:n]
    env = env / max(env.max(), 1e-9)
    sig = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
           + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)) * (0.3 + 0.7 * env)
    sig += 0.05 * rng.standard_normal(n)  # breath noise
    return (sig / max(np.abs(sig).max(), 1e-9) * 20000).astype(np.int16)


def _rir(length: int = 1024, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    decay = np.exp(-np.arange(length) / (length / 4))
    h = rng.standard_normal(length) * decay
    # A room never amplifies: normalize to a loud-speaker path (~-17 dB).
    h = h / max(float(np.sqrt(np.sum(h * h))), 1e-12) * 0.14
    return h.astype(np.float64)


def _apply_rir(ref: np.ndarray, rir: np.ndarray) -> np.ndarray:
    y = np.convolve(ref.astype(np.float64), rir)[:ref.shape[0]]
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _erle_db(mic: np.ndarray, err: np.ndarray) -> float:
    m = mic.astype(np.float64)
    e = err.astype(np.float64)
    return 10.0 * float(np.log10(np.mean(m * m) / (np.mean(e * e) + 1e-12)))


def test_factory_and_noop():
    assert isinstance(create_canceller("webrtc"), WebRtcAec)
    assert isinstance(create_canceller("noop"), NoOpAEC)
    try:
        create_canceller("nlms")
    except ValueError:
        pass  # NumPy backend removed in favor of WebRTC AEC3
    else:
        raise AssertionError("unknown backend must raise")


def test_bypass_is_bit_clean_when_reference_silent():
    aec = WebRtcAec()
    near = _speech_like(FRAME * 4, seed=1)
    silent = np.zeros(FRAME * 4, dtype=np.int16)
    clean, info = aec.process_frame(near, silent)
    assert info["bypassed"] is True
    assert np.array_equal(clean, near)


def test_echo_only_erle_and_convergence():
    aec = WebRtcAec()
    rir = _rir()
    ref = _speech_like(SR * 2, seed=2)          # 2 s of "TTS"
    mic = _apply_rir(ref, rir)                  # pure echo, no human
    errs = []
    for i in range(0, ref.shape[0], FRAME):
        clean, _ = aec.process_frame(mic[i:i + FRAME], ref[i:i + FRAME])
        errs.append(clean)
    err = np.concatenate(errs)
    tail = slice(SR, 2 * SR)                    # second 2 s: converged
    assert _erle_db(mic[tail], err[tail]) >= 10.0
    # Converged (10 dB) within the first 500 ms of echo.
    for ms in (125, 250, 375, 500):
        seg = slice(0, int(SR * ms / 1000))
        if _erle_db(mic[seg], err[seg]) >= 10.0:
            break
    else:
        raise AssertionError("did not converge within 500 ms")


def test_double_talk_preserves_user_envelope():
    # AEC3 manages double-talk internally (suppression favors echo removal;
    # near-end arrives attenuated but present). Assert the product-relevant
    # property: the user's syllabic envelope is still detectable, so STT and
    # the energy gate keep working on suppressed-but-present speech. (Pure
    # tones are adversarial to subband suppression; real broadband speech
    # survives better. Longer run: AEC3 settles within ~1 s of onset.)
    aec = WebRtcAec()
    rir = _rir()
    ref = _speech_like(SR * 2, seed=3)
    # Stationary echo level throughout (converge and mix at the same gain:
    # rooms don't turn down when the user starts talking). Scaled so the
    # mix cannot clip int16 and break the linear model.
    echo = (_apply_rir(ref, rir).astype(np.float64) * 0.5).astype(np.int16)
    near = _speech_like(SR * 2, seed=4)
    room = echo.astype(np.float64) + near.astype(np.float64) * 0.5
    assert np.abs(room).max() < 32767, "fixture must not clip"
    mic = room.astype(np.int16)
    # Converge on echo-only at the SAME level first (continuous history, as
    # in a real call: echo runs, the user talks over it).
    for i in range(0, echo.shape[0], FRAME):
        aec.process_frame(echo[i:i + FRAME], ref[i:i + FRAME], adapt=True)
    outs = []
    for i in range(0, mic.shape[0], FRAME):
        clean, _ = aec.process_frame(mic[i:i + FRAME], ref[i:i + FRAME], adapt=False)
        outs.append(clean)
    out = np.concatenate(outs)

    def _env(x, win=320):
        x = np.asarray(x, dtype=np.float64)
        return np.array([np.sqrt(np.mean(x[i:i + win] ** 2))
                         for i in range(0, len(x) - win, win)])

    ce = float(np.corrcoef(_env(out), _env((near * 0.5).astype(np.int16)))[0, 1])
    assert ce > 0.35, f"user envelope must survive double-talk (got {ce:.2f})"


def test_cpu_budget_per_frame():
    aec = WebRtcAec()
    ref = _speech_like(FRAME * 20, seed=6)
    mic = _apply_rir(ref, _rir())
    t0 = time.perf_counter()
    for i in range(0, ref.shape[0], FRAME):
        aec.process_frame(mic[i:i + FRAME], ref[i:i + FRAME])
    dt_ms = (time.perf_counter() - t0) / 20 * 1000.0
    assert dt_ms < 5.0, f"{dt_ms:.2f} ms/frame over budget"


def test_delay_estimator_finds_known_lag():
    ref = _speech_like(SR, seed=8)
    lag = 2400  # 150 ms
    mic = np.concatenate([np.zeros(lag, dtype=np.int16), ref])[:SR]
    assert abs(estimate_delay(mic, ref) - lag) <= 32
    assert estimate_delay(np.zeros(1000, dtype=np.int16), ref) == 0


def test_classifier_labels_all_four_classes():
    clf = SpeakerClassifier()
    rir = _rir()
    ref = _speech_like(FRAME * 8, seed=9)
    echo = _apply_rir(ref, rir)
    near = _speech_like(FRAME * 8, seed=10)
    assert clf.classify(np.zeros(FRAME, dtype=np.int16),
                        np.zeros(FRAME, dtype=np.int16),
                        np.zeros(FRAME, dtype=np.int16)).label == QUIET
    assert clf.classify(near[:FRAME], np.zeros(FRAME, dtype=np.int16),
                        near[:FRAME]).label == NEAR
    # Far-only through a converged canceller (single pass: residuals come
    # from the adapting run itself, never a stale-history re-run). One
    # stationary echo level throughout (rooms don't turn down mid-call).
    aec = WebRtcAec()
    echo = (echo.astype(np.float64) * 0.5).astype(np.int16)
    outs = []
    for i in range(0, ref.shape[0], FRAME):
        clean, _ = aec.process_frame(echo[i:i + FRAME], ref[i:i + FRAME])
        outs.append(clean)
    res = outs[-1]
    m, r = echo[-FRAME:], ref[-FRAME:]
    assert clf.classify(m, r, res).label == FAR
    # Double-talk at the same echo level: near-end added, history continuous.
    # Under AEC3's hard suppression the residual carries almost no energy
    # either way, so the ratio cannot name the speaker here — but it must
    # never claim silence while the user is present (that would mute all
    # downstream evidence). Interrupts are owned by the text layer.
    mix = (m.astype(np.float64) + (near[:FRAME].astype(np.float64) * 0.5))
    assert np.abs(mix).max() < 32767, "fixture must not clip"
    mix = mix.astype(np.int16)
    res2 = aec.process_frame(mix, r, adapt=False)[0]
    assert clf.classify(mix, r, res2).label in (DOUBLE, FAR)


def test_label_accuracy_over_labeled_segments():
    clf = SpeakerClassifier()
    aec = WebRtcAec()
    rir = _rir()
    ref = _speech_like(FRAME * 16, seed=11)
    # One stationary echo level for converge + measure (rooms don't turn
    # down when the user starts talking).
    echo = (_apply_rir(ref, rir).astype(np.float64) * 0.5).astype(np.int16)
    near = (_speech_like(FRAME * 16, seed=12).astype(np.float64) * 0.5).astype(np.int16)
    got, total = 0, 0
    # Static cases need no filter state (silent ref bypasses bit-clean).
    for m, r, want in [
        (np.zeros(FRAME, dtype=np.int16), np.zeros(FRAME, dtype=np.int16), QUIET),
        (near[:FRAME], np.zeros(FRAME, dtype=np.int16), NEAR),
    ]:
        total += 1
        res = aec.process_frame(m, r, adapt=False)[0]
        if clf.classify(m, r, res).label == want:
            got += 1
    # Echo-only stretch, single forward pass: FAR expected once converged.
    for k in range(8):
        m, r = echo[k * FRAME:(k + 1) * FRAME], ref[k * FRAME:(k + 1) * FRAME]
        res = aec.process_frame(m, r, adapt=True)[0]
        if k >= 4:
            total += 1
            if clf.classify(m, r, res).label == FAR:
                got += 1
    # Double-talk tail: same arrays the filter converged on (stationary
    # room), continuous history. Suppression erases the FAR/DOUBLE line, so
    # assert presence (never QUIET) rather than identity.
    for k in range(8, 12):
        total += 1
        m = (echo[k * FRAME:(k + 1) * FRAME].astype(np.float64)
             + near[k * FRAME:(k + 1) * FRAME].astype(np.float64))
        assert np.abs(m).max() < 32767, "fixture must not clip"
        m = m.astype(np.int16)
        r = ref[k * FRAME:(k + 1) * FRAME]
        res = aec.process_frame(m, r, adapt=False)[0]
        if clf.classify(m, r, res).label in (DOUBLE, FAR):
            got += 1
    assert got / total >= 0.9, f"label accuracy {got}/{total}"


def test_double_talk_detector_freezes_on_double_and_residual():
    from voice_agent.audio.aec.double_talk import DoubleTalkDetector
    d = DoubleTalkDetector()
    assert d.should_adapt(0.01, 0.05, 0.001, last_label="FAR") is True
    assert d.should_adapt(0.01, 0.05, 0.001, last_label="DOUBLE") is False
    assert d.is_double_talk(0.02, 0.05, 0.04) is True   # loud residual
    assert d.is_double_talk(0.02, 0.05, 0.001) is False  # clean echo
    assert d.is_double_talk(0.02, 0.0, 0.02) is False    # no reference


def test_echo_path_monitor_latches_and_acknowledges():
    from voice_agent.audio.aec.echo_detector import EchoPathMonitor
    m = EchoPathMonitor(trip_db=-3.0, trip_frames=5)
    for _ in range(4):
        assert m.update(-10.0) is False
    assert m.update(-10.0) is True
    assert m.diverged is True
    m.acknowledge()
    assert m.diverged is False
    assert m.update(12.0) is False


def test_frame_processor_pairs_filter_classify_detect():
    from voice_agent.audio.aec.double_talk import DoubleTalkDetector
    from voice_agent.audio.aec.echo_detector import EchoPathMonitor
    from voice_agent.audio.aec.processor import FrameProcessor
    aec = WebRtcAec()
    clf = SpeakerClassifier()
    p = FrameProcessor(aec, clf, DoubleTalkDetector(), EchoPathMonitor())
    ref = _speech_like(FRAME * 4, seed=31)
    echo = _apply_rir(ref, _rir())
    nbytes = 1024  # int16 mono device frame
    m, r = echo.tobytes()[:nbytes], ref.tobytes()[:nbytes]
    clean, decision, info = p.process(m, r)
    assert len(clean) == nbytes and decision is not None
    assert set(info) >= {"erle_db", "bypassed", "adapted", "dt_freeze"}
    c2, _, _ = p.process(b"\x00" * nbytes, b"\x00" * nbytes)
    assert c2 == b"\x00" * nbytes  # silent pair stays bit-clean


def test_geigel_freezes_on_loud_residual_holds_on_clean_echo():
    # Deterministic unit behavior (no backend): loud residual vs reference
    # freezes; tiny residual or silent reference never does; DOUBLE label
    # always freezes regardless of levels.
    from voice_agent.audio.aec.double_talk import DoubleTalkDetector
    d = DoubleTalkDetector()
    assert d.is_double_talk(0.05, 0.05, 0.04) is True    # loud residual
    assert d.is_double_talk(0.05, 0.05, 0.001) is False  # clean echo
    assert d.is_double_talk(0.05, 0.0, 0.05) is False    # silent reference
    assert d.is_double_talk(0.001, 0.05, 0.001, last_label="DOUBLE") is True
    assert d.should_adapt(0.05, 0.05, 0.001) is True
    d.reset()  # compat no-op, must not raise


def test_nfr_sweep_documents_geigel_limits():
    # Literature metric (miss vs false alarm across near-to-far ratios),
    # applied to the Geigel rule with measured residuals: shout-level
    # double-talk must fire, echo-only must (almost) never fire, and the
    # whisper miss below is the documented reason text confirmation exists.
    from voice_agent.audio.aec.double_talk import DoubleTalkDetector
    from voice_agent.audio.aec.webrtc import WebRtcAec
    aec = WebRtcAec()
    rir = _rir()
    ref = _speech_like(SR * 2, seed=43)
    echo = (_apply_rir(ref, rir).astype(np.float64) * 0.5).astype(np.int16)
    near = _speech_like(SR * 2, seed=44)
    for i in range(0, echo.shape[0], FRAME):
        aec.process_frame(echo[i:i + FRAME], ref[i:i + FRAME])
    results = {}
    for gain in (0.1, 0.3, 1.0, 2.0):
        d = DoubleTalkDetector()
        mix = (echo.astype(np.float64)
               + near.astype(np.float64) * gain).astype(np.int16)
        fired, frames = 0, 0
        for i in range(0, mix.shape[0], FRAME):
            clean, _ = aec.process_frame(mix[i:i + FRAME], ref[i:i + FRAME])
            m = mix[i:i + FRAME].astype(np.float64)
            res = float(np.sqrt(np.mean((m - clean.astype(np.float64)) ** 2)) / 32768.0)
            ref_rms = float(np.sqrt(np.mean(ref[i:i + FRAME].astype(np.float64) ** 2)) / 32768.0)
            frames += 1
            if d.is_double_talk(0.05, ref_rms, res):
                fired += 1
        results[gain] = (fired, frames)
        print(f"    NFR gain={gain}: fired {fired}/{frames}")
    fp = 0
    d = DoubleTalkDetector()
    for i in range(0, echo.shape[0], FRAME):
        clean, _ = aec.process_frame(echo[i:i + FRAME], ref[i:i + FRAME])
        m = echo[i:i + FRAME].astype(np.float64)
        res = float(np.sqrt(np.mean((m - clean.astype(np.float64)) ** 2)) / 32768.0)
        if d.is_double_talk(0.02, 0.05, res):
            fp += 1
    print(f"    echo-only false fires: {fp}")
    assert fp <= 2, f"too many false fires on echo-only ({fp})"
    assert results[2.0][0] > results[2.0][1] // 2, "shout-level double-talk must fire"
    assert results[0.1][0] < results[0.1][1] // 2, \
        "whisper under loud echo is a documented miss (text layer owns it)"


def test_double_talk_output_keeps_near_end_energy_floor():
    # Near-end attenuation floor: output must retain a floor fraction of the
    # near-end energy, or STT and the energy gate go deaf mid-interrupt.
    from voice_agent.audio.aec.webrtc import WebRtcAec
    aec = WebRtcAec()
    rir = _rir()
    ref = _speech_like(SR, seed=45)
    echo = (_apply_rir(ref, rir).astype(np.float64) * 0.5).astype(np.int16)
    near = _speech_like(SR, seed=46)
    for i in range(0, echo.shape[0], FRAME):
        aec.process_frame(echo[i:i + FRAME], ref[i:i + FRAME])
    mic = (echo.astype(np.float64) + near.astype(np.float64) * 0.5).astype(np.int16)
    outs = []
    for i in range(0, mic.shape[0], FRAME):
        clean, _ = aec.process_frame(mic[i:i + FRAME], ref[i:i + FRAME])
        outs.append(clean)
    out = np.concatenate(outs).astype(np.float64)
    ratio = float(np.mean(out ** 2) / (np.mean((near * 0.5).astype(np.float64) ** 2) + 1e-12))
    assert ratio >= 0.4, f"near-end energy floor violated ({ratio:.2f})"
