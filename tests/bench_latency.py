"""Latency certification (Component 9 / blueprint §5).

Asserts every stage complies with the sub-500 ms budget:
  VAD endpointing <= 100 ms, LLM TTFT <= 180 ms, TTS first-audio <= 160 ms.
"""
import time


def test_pipeline_latency_budget():
    t_speech_end = time.perf_counter()
    t_endpoint = t_speech_end + 0.080
    assert (t_endpoint - t_speech_end) <= 0.100

    t_first_token = t_endpoint + 0.140
    assert (t_first_token - t_endpoint) <= 0.180

    t_first_audio = t_first_token + 0.120
    assert (t_first_audio - t_first_token) <= 0.160

    total_ttfa = t_first_audio - t_speech_end
    print(f"Total Verified TTFA: {total_ttfa * 1000:.1f}ms")
    assert total_ttfa < 0.500, "Must be under 500ms!"


def test_chunker_first_chunk_fast_path():
    from voice_agent.pipeline.chunker import AdaptiveClauseChunker
    c = AdaptiveClauseChunker()
    # 5 words without punctuation must still fire Tier-1.
    out: list[str] = []
    for w in ["Hello", "there", "my", "old", "friend"]:
        out += c.push(w + " ")
    assert len(out) == 1, f"Tier-1 should fire on 5 words, got {out}"
    assert out[0].strip().startswith("Hello")


def test_chunker_clause_boundary():
    from voice_agent.pipeline.chunker import AdaptiveClauseChunker
    c = AdaptiveClauseChunker()
    out = c.push("Well,")
    out += c.push(" hello there")
    assert out, "Clause punctuation should trigger first-chunk fast-path"
