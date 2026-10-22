"""LearnedTurnDetector + EnergyVAD + DecisionPolicy on a tiny random checkpoint; tiny training run."""

from pathlib import Path

import numpy as np
import pytest
import torch

from turnsense.audio.frames import FrameBuffer
from turnsense.audio.vad import EnergyVAD
from turnsense.detectors.base import DetectorInput
from turnsense.detectors.learned import LearnedTurnDetector, read_checkpoint_metadata
from turnsense.events import DecisionType, DetectorMode, PolicyConfig, TurnEventType
from turnsense.policy import DecisionPolicy
from turnsense.research.evaluate import stream_recording
from turnsense.research.features import FeatureConfig
from turnsense.research.model import ModelConfig, TurnNet
from turnsense.research.train import TrainConfig, save_turn_checkpoint
from turnsense.sim.voice import VoiceProfile, noise_bed, synth_phrase


def _checkpoint(tmp_path: Path, mode: str, bias_shift: float | None = None) -> Path:
    torch.manual_seed(0)
    net = TurnNet(ModelConfig())
    if bias_shift is not None:  # make P(turn end) high whenever the model is not sure of speech
        with torch.no_grad():
            net.head.bias.copy_(torch.tensor([0.0, 0.0, bias_shift]))
    path = tmp_path / f"{mode}.pt"
    save_turn_checkpoint(path, net, mode, FeatureConfig(), TrainConfig(beta=1.0 if mode == "anti_transfer" else 0.0),
                         dataset_id="test", split_id="split-test", synthetic=True)  # fmt: skip
    return path


def _audio() -> np.ndarray:
    rng = np.random.default_rng(1)
    v = VoiceProfile.from_id(5)
    sp = synth_phrase(v, 6, "final", rng)
    audio = np.concatenate([np.zeros(8000, np.float32), sp, np.zeros(48_000, np.float32)])
    return audio + noise_bed(audio.size, -62.0, rng)


def test_metadata_and_mode_checks(tmp_path):
    path = _checkpoint(tmp_path, "anti_transfer")
    meta = read_checkpoint_metadata(path)
    assert meta["model_version"].startswith("turn-detector-at-0.1.0+")
    assert meta["synthetic"] is True and meta["beta"] == 1.0
    det = LearnedTurnDetector(DetectorMode.ANTI_TRANSFER, path, PolicyConfig())
    assert det.model_version == meta["model_version"] and det.probabilistic
    with pytest.raises(ValueError):
        LearnedTurnDetector(DetectorMode.LEARNED_CONTROL, path, PolicyConfig())
    exp = _checkpoint(tmp_path, "speaker_adversarial")
    with pytest.raises(ValueError):
        LearnedTurnDetector(DetectorMode.LEARNED_CONTROL, exp, PolicyConfig())
    LearnedTurnDetector(DetectorMode.LEARNED_CONTROL, exp, PolicyConfig(), allow_experimental=True)


def test_streaming_detector_commits_turn_through_policy(tmp_path):
    path = _checkpoint(tmp_path, "learned_control", bias_shift=6.0)
    policy_cfg = PolicyConfig()
    det = LearnedTurnDetector(DetectorMode.LEARNED_CONTROL, path, policy_cfg)
    vad, policy = EnergyVAD(), DecisionPolicy(policy_cfg)
    events, commits = [], []
    for frame in FrameBuffer().push(_audio()):
        v = vad.process(frame)
        out = det.process(DetectorInput(frame, v, agent_speaking=False))
        events.append(out)
        dec = policy.update(out, v, agent_active=False)
        if dec and dec.decision == DecisionType.TURN_COMPLETE:
            commits.append(dec.last_user_frame_ms)
            det.new_turn()
    kinds = {e.event_type for e in events}
    assert TurnEventType.TURN_COMPLETE in kinds
    assert len(commits) == 1  # new_turn() stops repeated commits during the same silence
    pause_events = [e for e in events if e.event_type != TurnEventType.CONTINUE]
    assert all(e.confidence is not None and e.p_turn_end is not None for e in pause_events)
    assert max(e.inference_ms for e in events) < 200


def test_barge_in_uses_shared_overlap_logic(tmp_path):
    det = LearnedTurnDetector(
        DetectorMode.LEARNED_CONTROL, _checkpoint(tmp_path, "learned_control"), PolicyConfig()
    )
    vad = EnergyVAD()
    saw_barge_in = False
    for frame in FrameBuffer().push(_audio()):
        out = det.process(DetectorInput(frame, vad.process(frame), agent_speaking=True))
        saw_barge_in |= out.event_type == TurnEventType.BARGE_IN
    assert saw_barge_in


def test_stream_recording_matches_manual_loop(tmp_path):
    path = _checkpoint(tmp_path, "learned_control", bias_shift=6.0)
    det = LearnedTurnDetector(DetectorMode.LEARNED_CONTROL, path, PolicyConfig())
    commits, inference, n_frames = stream_recording(_audio(), det, PolicyConfig())
    assert len(commits) == 1 and n_frames == len(inference) == _audio().size // 320
    commits2, _, _ = stream_recording(_audio(), det, PolicyConfig())  # reset() between runs
    assert commits == commits2


def test_tiny_training_run_produces_loadable_checkpoint(tmp_path):
    from turnsense.research.data import build_splits, load_manifest, prepare
    from turnsense.research.synthetic import SyntheticConfig, generate_corpus
    from turnsense.research.train import SpeakerTrainConfig, train_speaker_net, train_turn_net

    cfg = SyntheticConfig(n_train_speakers=2, n_unseen_speakers=2, convs_per_train_speaker=2,
                          convs_per_unseen_speaker=1, turns_per_conv=(2, 2))  # fmt: skip
    manifest, unseen, info = generate_corpus(cfg, tmp_path / "syn")
    assert info["SYNTHETIC"] is True
    recs = load_manifest(manifest)
    split = build_splits(recs, val_frac=0.25, unseen_speakers=unseen)
    prepared = prepare(recs, FeatureConfig())
    train = [p for p in prepared if split.assignments[p.rec.recording_id] == "train"]
    val = [p for p in prepared if split.assignments[p.rec.recording_id] == "val_seen"]
    mcfg = ModelConfig()
    spk, rep = train_speaker_net(train, val, mcfg, FeatureConfig(),
                                 SpeakerTrainConfig(epochs=1, steps_per_epoch=2, batch_size=4), log=lambda m: None)  # fmt: skip
    assert rep["n_speakers"] == 2
    tcfg = TrainConfig(epochs=1, steps_per_epoch=2, batch_size=2, crop_s=2.0, beta=1.0)
    net, trep = train_turn_net(train, val, mcfg, FeatureConfig(), tcfg, spk, log=lambda m: None)
    assert trep["history"][0]["train_at"] is not None
    path = tmp_path / "anti_transfer.pt"
    save_turn_checkpoint(path, net, "anti_transfer", FeatureConfig(), tcfg,
                         dataset_id=cfg.dataset_id, split_id=split.split_id, synthetic=True)  # fmt: skip
    det = LearnedTurnDetector(DetectorMode.ANTI_TRANSFER, path, PolicyConfig())
    assert det.info()["split_id"] == split.split_id
