"""Streaming == batch for features and model; pitch sanity."""

import numpy as np
import pytest
import torch

from turnsense.research.features import FeatureConfig, LogMelPitch, StreamingFeatures
from turnsense.research.model import HOLD, SHIFT, ModelConfig, SpeakerNet, TurnNet


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(24_000) / 16_000
    sig = 0.2 * np.sin(2 * np.pi * 150 * t) * (t < 0.8) + 0.02 * rng.standard_normal(t.size)
    return sig.astype(np.float32)


def test_streaming_features_equal_batch(audio):
    fe = LogMelPitch()
    batch = fe.batch(audio)
    stream = StreamingFeatures(fe)
    parts = [stream.push(audio[i : i + 320]) for i in range(0, audio.size - 319, 320)]
    streamed = np.concatenate(parts)
    assert batch.shape == streamed.shape == (2 * (audio.size // 320), fe.dim)
    # exactly two feature windows complete per 20 ms frame, from the very first frame
    assert all(p.shape[0] == 2 for p in parts)
    np.testing.assert_allclose(batch, streamed, atol=1e-5)


def test_streaming_features_odd_chunks(audio):
    fe = LogMelPitch()
    stream = StreamingFeatures(fe)
    sizes = [7, 313, 640, 1, 999, 2000]
    out, i = [], 0
    while i < audio.size:
        n = sizes[len(out) % len(sizes)]
        out.append(stream.push(audio[i : i + n]))
        i += n
    streamed = np.concatenate(out)
    full = fe.batch(audio)
    np.testing.assert_allclose(streamed[: full.shape[0]], full, atol=1e-5)


@pytest.mark.parametrize("f0", [90.0, 150.0, 240.0])
def test_pitch_feature_tracks_tone(f0):
    fe = LogMelPitch(FeatureConfig())
    t = np.arange(16_000) / 16_000
    tone = (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    feats = fe.batch(tone)
    est = 100 * 2 ** np.median(feats[10:90, -1])
    assert abs(est - f0) / f0 < 0.03
    assert np.median(feats[10:90, -2]) > 0.9  # strongly periodic


def test_model_step_equals_forward():
    torch.manual_seed(0)
    net = TurnNet(ModelConfig()).eval()
    x = torch.randn(2, 61, 42)
    with torch.no_grad():
        full, maps = net(x)
        state = net.init_state(2)
        outs, i = [], 0
        for n in [1, 2, 3, 2, 7, 2, 44]:
            o, state = net.step(x[:, i : i + n], state)
            outs.append(o)
            i += n
    assert i == x.shape[1]
    torch.testing.assert_close(full, torch.cat(outs, 1), atol=1e-5, rtol=1e-4)
    assert set(maps) == {"conv1", "conv2", "conv3", "gru"}
    assert maps["conv3"].shape == (2, 64, 61)


def test_p_turn_end_is_conditional_probability():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    p = TurnNet.p_turn_end(logits)
    assert torch.all((p >= 0) & (p <= 1))
    assert p[1] < 0.5 < p[2]
    assert abs(float(p[0]) - 0.5) < 1e-4  # speech-dominant: hold vs shift undecided
    assert HOLD == 1 and SHIFT == 2


def test_speaker_net_shares_encoder_architecture():
    cfg = ModelConfig()
    a, b = TurnNet(cfg), SpeakerNet(cfg, 7)
    shapes_a = {k: v.shape for k, v in a.encoder.state_dict().items()}
    shapes_b = {k: v.shape for k, v in b.encoder.state_dict().items()}
    assert shapes_a == shapes_b
    logits, _ = b(torch.randn(3, 50, 42), torch.ones(3, 50, dtype=torch.bool))
    assert logits.shape == (3, 7)
