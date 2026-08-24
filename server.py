"""
Voice AI realtime server.

Flow:  Browser mic --WebRTC--> [VAD] --> faster-whisper (STT) --> Groq LLM
       --> Coqui TTS --> WAV --> WebRTC audio track back to the browser.
"""

# asyncio powers the event loop that aiortc and FastAPI both run on,
# and to_thread offloads the blocking CPU work (whisper/TTS) so audio keeps flowing.
import asyncio

# collections.deque is an O(1) FIFO queue used to buffer outgoing audio frames.
import collections

# fractions.Fraction is required for AV frame timestamps (PyAV 17 no longer
# re-exports it as av.Fraction).
from fractions import Fraction

# time gives us a monotonic clock for pacing outbound audio at real-time speed.
import time

# PyAV ships with aiortc; AudioResampler converts between sample rates/layouts,
# and av.open lets us decode the TTS WAV bytes into raw frames in memory.
import av

# numpy handles fast slicing/reshaping of interleaved PCM samples.
import numpy as np

# FastAPI is the web framework serving the page, SDP signaling and WS events.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# StaticFiles mounts the static/ folder; FileResponse serves index.html at /.
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# pydantic BaseModel validates the JSON body of POST /offer.
from pydantic import BaseModel

# aiortc is the actual WebRTC implementation (peer connections, media tracks).
from aiortc import RTCPeerConnection, RTCSessionDescription

# MediaStreamTrack is the base class we subclass to build our audio "player".
# MediaStreamError is raised by inbound tracks when the browser disconnects.
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError

# Our own pipeline pieces built earlier in src/.
from src.vad import VADetector, FRAME_BYTES          # speech start/end detection
from src.stt import transcribe                        # PCM -> text
from src.llm_client import chat                       # text -> reply text
from src.tts_engine import synthesize                 # text -> WAV bytes

# Create the app instance uvicorn will serve on http://localhost:8000.
app = FastAPI(title="VoiceAI")

# Keep every peer connection in a set so we can close them cleanly on shutdown.
pcs = set()

# Registry of connected WebSockets so transcripts/status can be broadcast.
clients = set()

# Global "busy" flag: while True the mic stream is ignored so the AI isn't
# interrupted by its own voice coming through the user's speakers.
busy = False


# ---------- outbound audio player -------------------------------------------

class PlayerTrack(MediaStreamTrack):
    """A push-based audio track: pipeline code drops WAV data in, aiortc pulls frames out."""

    # Tells aiortc this track carries audio (not video).
    kind = "audio"

    def __init__(self):
        # Initialize parent class internals (id, stopped flag, etc.).
        super().__init__()
        # Queue of ready-to-send 48kHz stereo s16 numpy chunks.
        self._queue = collections.deque()
        # Wall-clock reference point when playback of a burst begins.
        self._start = None
        # Next absolute deadline for sending a frame (keeps real-time pacing).
        self._next_time = None
        # Monotonically increasing RTP timestamp measured in samples @48kHz.
        self._pts = 0
        # Frames are 1024 samples/channel (~21ms) - small enough for low jitter,
        # large enough to keep per-frame overhead negligible.
        self.samples_per_frame = 1024
        self.rate = 48000

    async def recv(self):
        """Called by aiortc whenever it needs the next frame; we pace to real time."""

        now = time.monotonic()

        if self._start is None:
            # First frame ever: anchor the pacing clock to right now.
            self._start = now
            self._next_time = now
        else:
            # If we're ahead of schedule, sleep so audio plays at natural speed
            # instead of blasting the whole reply in a fraction of a second.
            wait = self._next_time - now
            if wait > 0:
                await asyncio.sleep(wait)

        # Schedule the next frame exactly one frame-duration later.
        self._next_time += self.samples_per_frame / self.rate

        if len(self._queue) > 0:
            # Real audio available -> pop it.
            data = self._queue.popleft()
        else:
            # Nothing queued -> send digital silence, which keeps the RTP stream
            # alive and lets the browser's jitter buffer stay stable.
            # Shape must be (1, N) because PyAV packed s16 expects one row of
            # interleaved samples: N = channels(2) * samples_per_frame.
            data = np.zeros((1, self.samples_per_frame * 2), dtype=np.int16)

        # Build an AV audio frame from the int16 array of shape (1, interleaved).
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(data), format="s16", layout="stereo"
        )
        frame.sample_rate = self.rate
        # Timestamp every frame so receivers can order/jitter-buffer correctly.
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.rate)
        self._pts += self.samples_per_frame
        return frame

    def push_wav(self, wav_bytes: bytes):
        """Decode TTS WAV bytes and enqueue them as paced 48k stereo frames."""

        # Open the in-memory WAV as an audio container to access encoded packets.
        container = av.open(io_bytes(wav_bytes))
        # Resampler converts 22050Hz mono -> 48kHz stereo s16 in one call;
        # AV automatically upmixes mono->stereo during resampling.
        resampler = av.AudioResampler(format="s16", layout="stereo", rate=48000)

        flat = []
        for packet in container.decode(audio=0):
            # resample() returns a list of converted AudioFrames per input packet.
            for f in resampler.resample(packet):
                # Packed s16 -> shape (1, N*2); flatten to one interleaved list.
                flat.append(f.to_ndarray().reshape(-1))

        # Concatenate everything into a single interleaved [L,R,L,R,...] buffer.
        pcm = np.concatenate(flat).astype(np.int16)

        # Slice into fixed frames of 2*1024 interleaved samples (stereo!),
        # skipping any trailing partial frame - a few ms cut at the very
        # end is inaudible.
        n = self.samples_per_frame * 2
        count = len(pcm) // n
        for i in range(count):
            chunk = pcm[i * n:(i + 1) * n]
            # Keep interleaved layout in a single row, which is exactly the
            # shape av.AudioFrame.from_ndarray(format="s16") requires.
            self._queue.append(chunk.reshape(1, -1))

    def pending_duration(self) -> float:
        """Seconds of unplayed audio still queued - used to know when playback ends."""
        return len(self._queue) * self.samples_per_frame / self.rate


def io_bytes(data: bytes):
    """Small helper wrapping bytes in BytesIO (kept separate for readability)."""
    import io
    return io.BytesIO(data)


# ---------- transcript broadcast helpers ------------------------------------

async def broadcast(payload: dict):
    """Send a JSON status/transcript message to every connected browser tab."""
    for ws in list(clients):
        try:
            await ws.send_json(payload)
        except Exception:
            # A dead websocket must never break the audio pipeline.
            clients.discard(ws)


# ---------- the per-utterance pipeline --------------------------------------

async def run_turn(utterance: bytes, player: PlayerTrack):
    """STT -> LLM -> TTS for one detected utterance; runs while mic is muted."""

    global busy

    try:
        await broadcast({"type": "status", "state": "transcribing"})

        # whisper is pure CPU blocking work -> run it in a worker thread
        # so the asyncio loop keeps serving WebRTC frames meanwhile.
        text = await asyncio.to_thread(transcribe, utterance)

        if not text:
            # Whisper heard nothing intelligible -> skip the whole turn silently.
            return

        await broadcast({"type": "user", "text": text})
        await broadcast({"type": "status", "state": "thinking"})

        # Groq API call is blocking HTTP -> also pushed to a thread.
        reply = await asyncio.to_thread(chat, text)
        await broadcast({"type": "ai", "text": reply})

        await broadcast({"type": "status", "state": "speaking"})

        # Tacotron2 inference is the slowest step (~seconds on CPU) -> thread.
        wav = await asyncio.to_thread(synthesize, reply)

        # Hand the finished WAV to the player track feeding the browser.
        player.push_wav(wav)

        # Wait until the queue drains so 'busy' stays True until the user has
        # actually heard the full answer (not just until generation ended).
        while player.pending_duration() > 0:
            await asyncio.sleep(0.05)
        # Small grace period covering the last buffered frame reaching the speaker.
        await asyncio.sleep(0.3)

    except Exception as e:
        # Never let one bad turn kill the connection - report and move on.
        await broadcast({"type": "status", "state": f"error: {e}"})

    finally:
        busy = False
        await broadcast({"type": "status", "state": "listening"})


# ---------- inbound mic consumer ---------------------------------------------

async def consume_mic(track, player: PlayerTrack):
    """Pull decoded mic frames off the WebRTC track, run VAD, trigger turns."""

    global busy

    # Resamples whatever the browser sends (Opus-decoded 48k stereo)
    # down to exactly what VAD+whisper need: 16kHz mono signed-16bit.
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    # Raw byte accumulator: resampled chunks arrive in odd sizes, but VAD
    # requires exact 30ms (960-byte) slices, so we buffer and slice ourselves.
    acc = bytearray()
    vad = VADetector()

    while True:
        try:
            # Await the next decoded audio frame from the network.
            frame = await track.recv()
        except MediaStreamError:
            # Browser disconnected/closed the track -> end this consumer task.
            return

        # Ignore user's mic entirely while the AI reply is playing (no barge-in v1),
        # otherwise the speakers' output picked up by the mic would re-trigger STT.
        if busy:
            continue

        # Resample returns a list (usually one) of converted mono 16k frames.
        for f in resampler.resample(frame):
            # Flatten packed s16 data to raw little-endian PCM bytes.
            acc.extend(f.to_ndarray().reshape(-1).tobytes())

        # Process every complete 30ms window that has accumulated.
        while len(acc) >= FRAME_BYTES:
            chunk = bytes(acc[:FRAME_BYTES])   # take exactly one VAD frame
            del acc[:FRAME_BYTES]              # remove it from the buffer
            utterance = vad.process(chunk)     # feed the state machine

            if utterance is not None and not busy:
                # Full utterance detected -> fire the pipeline without blocking
                # the mic loop (it keeps running/VAD-resetting meanwhile).
                busy = True
                asyncio.ensure_future(run_turn(utterance, player))


# ---------- HTTP endpoints ----------------------------------------------------

# Request schema for POST /offer: standard trickle-free SDP exchange body.
class Offer(BaseModel):
    sdp: str      # session description payload from the browser
    type: str     # always "offer"


@app.post("/offer")
async def offer(params: Offer):
    """WebRTC signaling: turn the browser's SDP offer into our SDP answer."""

    # One RTCPeerConnection per browser tab/session.
    pc = RTCPeerConnection()

    # Track it globally for cleanup on server shutdown.
    pcs.add(pc)

    # The player must be added BEFORE createAnswer() so the negotiated SDP
    # includes a send-direction audio m-line for the AI voice.
    player = PlayerTrack()
    pc.addTrack(player)

    @pc.on("track")
    def on_track(track):
        # Fires when the remote (browser) starts sending its mic track.
        if track.kind == "audio":
            # Start the async consumer that does resampling + VAD + turns.
            asyncio.ensure_future(consume_mic(track, player))

    @pc.on("connectionstatechange")
    async def on_state():
        # Clean up fully once the browser leaves or the link fails.
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    # Load the remote description received from the browser.
    await pc.setRemoteDescription(RTCSessionDescription(sdp=params.sdp, type=params.type))

    # Generate our side of the negotiation (answer with player track).
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # localDescription reflects the actual SDP after ICE gathering defaults,
    # which is what the browser must apply as its remote description.
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.websocket("/events")
async def events(ws: WebSocket):
    """WebSocket channel pushing transcripts/status updates to the UI."""

    # Complete the HTTP->WS upgrade handshake.
    await ws.accept()
    clients.add(ws)

    # Tell the new tab immediately where things stand.
    await ws.send_json({"type": "status", "state": "listening"})

    try:
        # We never expect client messages; just hold the socket open,
        # receiving (and ignoring) anything until the client disconnects.
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


@app.get("/")
async def index():
    # Serve the single-page client at the root URL.
    return FileResponse("static/index.html")


@app.on_event("shutdown")
async def shutdown():
    """Close all peer connections when Ctrl-C stops the server."""
    for pc in list(pcs):
        await pc.close()
