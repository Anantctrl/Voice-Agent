"""Ring-buffer saturation test (blueprint §6: non-blocking under saturation)."""
from voice_agent.audio.ring_buffer import PlaybackRingBuffer


def test_overwrite_oldest_on_saturation():
    rb = PlaybackRingBuffer(capacity_bytes=16)
    rb.write(b"A" * 16)
    assert len(rb) == 16
    rb.write(b"BC")  # overflow by 2 -> drop 2 oldest
    assert len(rb) == 16
    out = bytearray(16)
    rb.read(16, out)
    assert bytes(out) == b"A" * 14 + b"BC"


def test_clear_is_instant():
    rb = PlaybackRingBuffer(capacity_bytes=64)
    rb.write(b"x" * 64)
    rb.clear()
    assert len(rb) == 0
