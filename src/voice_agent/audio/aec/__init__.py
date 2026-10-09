"""AEC subpackage: WebRTC AEC3 backend + speaker attribution."""
from voice_agent.audio.aec.double_talk import DoubleTalkDetector
from voice_agent.audio.aec.echo_detector import EchoPathMonitor, estimate_delay
from voice_agent.audio.aec.nlms import NoOpAEC, create_canceller
from voice_agent.audio.aec.processor import FrameProcessor
from voice_agent.audio.aec.webrtc import WebRtcAec

__all__ = [
    "DoubleTalkDetector",
    "EchoPathMonitor",
    "FrameProcessor",
    "NoOpAEC",
    "WebRtcAec",
    "create_canceller",
    "estimate_delay",
]
