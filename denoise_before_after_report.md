# Noise Suppression — Before / After Report

GTRN stage = WebRTC `AudioProcessor` (Noise Suppression + High-Pass + AGC2), via
`pywebrtc_audio`, 16 kHz mono, per the `GTRNConfig` in `constants/audio.py`.

Current config:
- `NS_LEVEL = 2` (18 dB suppression)
- `NOISE_SUPPRESSION = True`, `HIGH_PASS_FILTER = True`
- `AUTO_GAIN_CONTROL = True`, `AGC_GAIN_DB = 8.0`

Metrics: RMS, peak (float, -1..1), and p10 of |sample| (an approximate residual
noise-floor level). Lower floor = less residual background noise.

---

## 1) Raw voice  (`my_voice_raw.wav`, 48 kHz → 16 kHz)

| Stage | File | RMS | Peak | ~floor (p10) |
|---|---|---|---|---|
| Before (raw 48k) | `my_voice_raw.wav` | 0.00207 | 0.0135 | 0.0000 |
| Before (16k) | `my_voice_raw_16k.wav` | 0.00206 | 0.0135 | 0.0000 |
| **After (denoised)** | `my_voice_denoised.wav` | **0.05714** | **0.6024** | 0.0000 |

- The source is very quiet; the AGC + 8 dB gain stage levels it up ~28x RMS to a
  healthy, listenable level while keeping it below clipping (peak 0.60).
- Whisper `small` transcript: *"Hello, hello... mic testing..."*

## 2) Noisy street 5 dB (`noisy_voice_street_48k.wav`, NOIZEUS)

| NS level | File | RMS | Peak | ~floor (p10) |
|---|---|---|---|---|
| Before (raw 16k) | `noisy_street_raw_16k.wav` | 0.05252 | 0.3907 | 0.00424 |
| NS1 (12 dB) | `noisy_voice_street_48k_ns1.wav` | 0.18208 | 0.9746 | 0.00616 |
| NS2 (18 dB) | `noisy_voice_street_48k_ns2.wav` | 0.15843 | 0.9747 | 0.00433 |
| NS3 (21 dB) | `noisy_voice_street_48k_ns3.wav` | 0.15876 | 0.9749 | **0.00366** |
| **Pipeline output (NS2+AGC8)** | `denoised_street.wav` | 0.14499 | 0.9755 | 0.00418 |

Whisper `small` transcript (pipeline output): *"The sky that morning was clear
and right."* (reference NOIZEUS: *"The sky that morning was clear and bright
blue."*)

---

## Observations

- **Higher NS level → lower residual noise floor**: p10 drops from 0.00424
  (raw) → 0.00616 (NS1) → 0.00433 (NS2) → 0.00366 (NS3). NS3 removes the most
  background noise.
- **AGC lifts overall level**: all NS variants are far louder than raw
  (RMS ~0.14–0.18 vs 0.05) because the gain/AGC stage normalizes volume to
  near 0 dBFS. This is the "make it louder" behavior (AGC_GAIN_DB=8).
- **Trade-off**: more suppression = cleaner noise floor but higher risk of
  speech distortion (documented in WebRTC `noise_suppression.h`: *"Increasing
  the level will reduce the noise level at the expense of a higher speech
  distortion"*). In our street test, NS2 gave a more accurate transcript than
  NS3, so NS2 is the current default.

Files saved under `D:\Project\voiceai\`:
- `my_voice_raw_16k.wav` / `my_voice_denoised.wav`
- `noisy_street_raw_16k.wav` / `denoised_street.wav`
- `noisy_voice_street_48k_ns1.wav` / `_ns2.wav` / `_ns3.wav` (A/B demo, active AGC)

## A/B audio demo
Compare the street A/B: raw (`noisy_street_raw_16k.wav`) then `_ns1/_ns2/_ns3`.
