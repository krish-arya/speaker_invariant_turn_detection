"""Live-path pieces that can be verified without provider credentials or a LiveKit server."""

import asyncio

import numpy as np
import pytest

from turnsense.adapters.deepgram import parse_stt_message
from turnsense.adapters.factory import ProviderConfigError, build_live_providers
from turnsense.adapters.openai_compat import parse_sse_line
from turnsense.config import Settings
from turnsense.worker.agent import FrameQueue, LiveKitSink, load_session_info

from .conftest import frame, make_info


def test_deepgram_results_mapping():
    msg = {
        "type": "Results",
        "is_final": False,
        "start": 2.0,
        "duration": 0.5,
        "channel": {"alternatives": [{"transcript": " book a table "}]},
    }
    ev = parse_stt_message(msg, offset_ms=100)
    assert ev.text == "book a table" and not ev.is_final
    assert (ev.start_ms, ev.end_ms) == (2100.0, 2600.0)
    assert parse_stt_message({"type": "Metadata"}) is None
    assert parse_stt_message({**msg, "channel": {"alternatives": [{"transcript": ""}]}}) is None


def test_sse_parsing():
    assert parse_sse_line('data: {"choices":[{"delta":{"content":"Hel"}}]}') == "Hel"
    assert parse_sse_line('data: {"choices":[{"delta":{}}]}') == ""
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line(": keep-alive") == ""


def test_hosted_profile_requires_keys_and_never_echoes_them(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path, deepgram_api_key=None, groq_api_key=None)
    with pytest.raises(ProviderConfigError) as err:
        build_live_providers(s)
    assert "DEEPGRAM_API_KEY" in str(err.value)
    s = Settings(_env_file=None, data_dir=tmp_path, DEEPGRAM_API_KEY="dg-secret", GROQ_API_KEY="gq-secret")
    assert "dg-secret" not in repr(s) and "gq-secret" not in s.model_dump_json()


def test_frame_queue_drops_oldest_when_overloaded():
    async def go():
        q = FrameQueue(max_frames=3)
        for i in range(1, 6):
            q.put(frame(i))
        return q.dropped, [(await q.get()).seq for _ in range(3)]

    assert asyncio.run(go()) == (2, [3, 4, 5])


class FakeSource:
    sample_rate = 24_000

    def __init__(self):
        self.frames = []
        self.cleared = 0

    async def capture_frame(self, f):
        self.frames.append(f)

    def clear_queue(self):
        self.cleared += 1

    async def wait_for_playout(self):
        return None


def test_livekit_sink_frames_and_resamples():
    src = FakeSource()
    sink = LiveKitSink(src)
    asyncio.run(sink.play(np.zeros(1600, np.float32), 16_000))  # 100 ms at 16 kHz → 24 kHz
    assert len(src.frames) == 10 and all(f.samples_per_channel == 240 for f in src.frames)
    sink.flush()
    assert src.cleared == 1


def test_session_info_lookup_rejects_unknown_and_malformed_ids(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert load_session_info(s, "../../etc/passwd") is None
    assert load_session_info(s, "ses_0123456789abcdef") is None
    s.sessions_dir.mkdir(parents=True)
    info = make_info()
    (s.sessions_dir / f"{info.session_id}.json").write_text(info.model_dump_json(), encoding="utf-8")
    assert load_session_info(s, info.session_id).trial_id == info.trial_id
