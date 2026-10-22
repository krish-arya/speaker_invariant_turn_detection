"""Training: orthogonal-task speaker network, then the turn model (control vs anti-transfer).

Controlled comparison (goal.md §6): ``learned_control`` (E1, β = 0) and ``anti_transfer``
(E2, β > 0) share the feature config, model config, seed, initial weights, batch sequence,
epochs, optimiser and model-selection rule. The only difference is the AT term. Model
selection uses the task loss on ``val_seen`` only — unseen speakers are never consulted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from turnsense.events import DetectorMode
from turnsense.research.anti_transfer import (
    SpeakerAdversary,
    anti_transfer_loss,
    gram_matrix,
    gram_similarity,
)
from turnsense.research.data import PreparedRecording
from turnsense.research.features import FeatureConfig
from turnsense.research.model import HOLD, IGNORE, SHIFT, SPEECH, ModelConfig, SpeakerNet, TurnNet

CHECKPOINT_FORMAT = "turnsense-turn-detector-1"
SPEAKER_FORMAT = "turnsense-speaker-net-1"
MODEL_VERSION_BASE = "0.1.0"
VARIANT_PREFIX = {
    DetectorMode.LEARNED_CONTROL.value: "lc",
    DetectorMode.ANTI_TRANSFER.value: "at",
    "speaker_adversarial": "sa",
}

Log = Callable[[str], None]


def _print(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class SpeakerTrainConfig:
    seed: int = 0
    epochs: int = 6
    steps_per_epoch: int | None = None
    batch_size: int = 32
    crop_s: float = 3.0
    lr: float = 2e-3
    weight_decay: float = 1e-4
    min_speech_frac: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 10
    steps_per_epoch: int | None = None
    batch_size: int = 16
    crop_s: float = 8.0
    warmup_frames: int = 30  # no loss on the first frames of a mid-recording crop
    lr: float = 2e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    class_weight_cap: float = 10.0
    beta: float = 0.0  # anti-transfer weight; 0 ⇒ learned_control
    at_layers: tuple[str, ...] = ("conv3",)
    at_similarity: str = "paper"
    adversary: str | None = None  # "grl" ⇒ speaker-adversarial ablation (not anti-transfer)
    grl_lambda: float = 0.5
    adversary_weight: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["at_layers"] = list(self.at_layers)
        return d


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class CropSampler:
    def __init__(
        self, data: Sequence[PreparedRecording], crop_frames: int, rng: np.random.Generator
    ) -> None:
        self.data = list(data)
        self.crop = crop_frames
        self.rng = rng
        lengths = np.array([d.features.shape[0] for d in self.data], dtype=np.float64)
        self.p = lengths / lengths.sum()
        self.dim = self.data[0].features.shape[1]

    def sample(
        self, batch: int, warmup: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        x = np.zeros((batch, self.crop, self.dim), dtype=np.float32)
        y = np.full((batch, self.crop), IGNORE, dtype=np.int64)
        mask = np.zeros((batch, self.crop), dtype=bool)
        idx = self.rng.choice(len(self.data), size=batch, p=self.p)
        for b, i in enumerate(idx):
            d = self.data[i]
            n_total = d.features.shape[0]
            start = (
                0 if n_total <= self.crop else int(self.rng.integers(0, n_total - self.crop + 1))
            )
            n = min(self.crop, n_total - start)
            x[b, :n] = d.features[start : start + n]
            y[b, :n] = d.labels[start : start + n]
            mask[b, :n] = True
            if start > 0 and warmup > 0:
                y[b, :warmup] = IGNORE
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), idx


def feature_stats(data: Sequence[PreparedRecording]) -> tuple[torch.Tensor, torch.Tensor]:
    n = 0
    s = None
    ss = None
    for d in data:
        f = d.features.astype(np.float64)
        s = f.sum(0) if s is None else s + f.sum(0)
        ss = (f**2).sum(0) if ss is None else ss + (f**2).sum(0)
        n += f.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 1e-8))
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def class_weights(data: Sequence[PreparedRecording], cap: float) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.float64)
    for d in data:
        for c in (SPEECH, HOLD, SHIFT):
            counts[c] += np.sum(d.labels == c)
    w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    w = np.minimum(w, cap)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _steps(cfg_steps: int | None, data: Sequence[PreparedRecording], batch: int, crop: int) -> int:
    if cfg_steps:
        return cfg_steps
    total = sum(d.features.shape[0] for d in data)
    return max(1, math.ceil(total / (batch * crop)))


# ---------------------------------------------------------------------------
# Speaker network (orthogonal task)
# ---------------------------------------------------------------------------


def train_speaker_net(
    train: Sequence[PreparedRecording],
    val: Sequence[PreparedRecording],
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    cfg: SpeakerTrainConfig | None = None,
    log: Log = _print,
) -> tuple[SpeakerNet, dict]:
    cfg = cfg or SpeakerTrainConfig()
    speakers = sorted({d.rec.speaker_id for d in train})
    spk_index = {s: i for i, s in enumerate(speakers)}
    torch.manual_seed(cfg.seed)
    net = SpeakerNet(model_cfg, len(speakers))
    net.encoder.set_normalisation(*feature_stats(train))
    rng = np.random.default_rng(cfg.seed + 101)
    crop = int(round(cfg.crop_s * feature_cfg.frames_per_second))
    by_spk: dict[str, list[PreparedRecording]] = {}
    for d in train:
        by_spk.setdefault(d.rec.speaker_id, []).append(d)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = _steps(cfg.steps_per_epoch, train, cfg.batch_size, crop)
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        net.train()
        losses, correct, seen = [], 0, 0
        for _ in range(steps):
            x, m, y = _speaker_batch(by_spk, speakers, spk_index, crop, cfg, rng)
            logits, _ = net(x, m)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += int((logits.argmax(1) == y).sum())
            seen += y.numel()
        val_acc = speaker_accuracy(net, val, spk_index, crop)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "train_acc": correct / seen, "val_seen_acc": val_acc})  # fmt: skip
        log(f"[speaker] epoch {epoch + 1}/{cfg.epochs} loss={np.mean(losses):.3f} "
            f"train_acc={correct / seen:.3f} val_seen_acc={val_acc:.3f}")  # fmt: skip
    net.eval()
    report = {
        "speakers": speakers,
        "n_speakers": len(speakers),
        "chance": 1.0 / len(speakers),
        "history": history,
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    return net, report


def _speaker_batch(by_spk, speakers, spk_index, crop, cfg: SpeakerTrainConfig, rng):
    dim = next(iter(by_spk.values()))[0].features.shape[1]
    x = np.zeros((cfg.batch_size, crop, dim), dtype=np.float32)
    m = np.zeros((cfg.batch_size, crop), dtype=bool)
    y = np.zeros(cfg.batch_size, dtype=np.int64)
    for b in range(cfg.batch_size):
        spk = speakers[int(rng.integers(len(speakers)))]
        recs = by_spk[spk]
        best = None
        for _ in range(6):  # prefer crops with enough speech to identify the voice
            d = recs[int(rng.integers(len(recs)))]
            n_total = d.features.shape[0]
            start = 0 if n_total <= crop else int(rng.integers(0, n_total - crop + 1))
            lab = d.labels[start : start + crop]
            frac = float(np.mean(lab == SPEECH)) if lab.size else 0.0
            if best is None or frac > best[2]:
                best = (d, start, frac)
            if frac >= cfg.min_speech_frac:
                break
        d, start, _ = best
        n = min(crop, d.features.shape[0] - start)
        x[b, :n] = d.features[start : start + n]
        m[b, :n] = d.labels[start : start + n] == SPEECH
        y[b] = spk_index[spk]
    return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y)


@torch.no_grad()
def speaker_accuracy(
    net: SpeakerNet, data: Sequence[PreparedRecording], spk_index: dict[str, int], crop: int
) -> float | None:
    net.eval()
    correct = total = 0
    for d in data:
        if d.rec.speaker_id not in spk_index:
            continue
        for start in range(0, d.features.shape[0] - crop + 1, crop):
            lab = d.labels[start : start + crop]
            if np.mean(lab == SPEECH) < 0.2:
                continue
            x = torch.from_numpy(d.features[None, start : start + crop])
            m = torch.from_numpy((lab == SPEECH)[None])
            logits, _ = net(x, m)
            correct += int(logits.argmax(1).item() == spk_index[d.rec.speaker_id])
            total += 1
    net.train()
    return correct / total if total else None


# ---------------------------------------------------------------------------
# Turn network (target task)
# ---------------------------------------------------------------------------


def train_turn_net(
    train: Sequence[PreparedRecording],
    val: Sequence[PreparedRecording],
    model_cfg: ModelConfig,
    feature_cfg: FeatureConfig,
    cfg: TrainConfig,
    speaker_net: SpeakerNet | None = None,
    log: Log = _print,
) -> tuple[TurnNet, dict]:
    if cfg.beta > 0 and speaker_net is None:
        raise ValueError("anti-transfer (beta > 0) needs a pretrained speaker network")
    if speaker_net is not None:
        speaker_net.eval()
        speaker_net.requires_grad_(False)

    torch.manual_seed(cfg.seed)
    net = TurnNet(model_cfg)
    net.encoder.set_normalisation(*feature_stats(train))
    adversary = None
    spk_index: dict[str, int] = {}
    if cfg.adversary == "grl":
        speakers = sorted({d.rec.speaker_id for d in train})
        spk_index = {s: i for i, s in enumerate(speakers)}
        torch.manual_seed(cfg.seed + 7)
        adversary = SpeakerAdversary(model_cfg.gru_hidden, len(speakers), cfg.grl_lambda)
    elif cfg.adversary is not None:
        raise ValueError(f"unknown adversary {cfg.adversary!r}")

    params = list(net.parameters()) + (list(adversary.parameters()) if adversary else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = class_weights(train, cfg.class_weight_cap)
    crop = int(round(cfg.crop_s * feature_cfg.frames_per_second))
    sampler = CropSampler(train, crop, np.random.default_rng(cfg.seed + 202))
    steps = _steps(cfg.steps_per_epoch, train, cfg.batch_size, crop)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps)

    best_state, best_val, best_epoch = None, math.inf, 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        net.train()
        ce_sum = at_sum = adv_sum = 0.0
        for _ in range(steps):
            x, y, mask, idx = sampler.sample(cfg.batch_size, cfg.warmup_frames)
            logits, maps = net(x)
            ce = F.cross_entropy(logits.reshape(-1, 3), y.reshape(-1), weight=weights,
                                 ignore_index=IGNORE)  # fmt: skip
            loss = ce
            if cfg.beta > 0:
                with torch.no_grad():
                    _, pre_maps = speaker_net.encoder(x)  # type: ignore[union-attr]
                at, _ = anti_transfer_loss(
                    maps, pre_maps, cfg.at_layers, cfg.beta, mask, cfg.at_similarity
                )  # type: ignore[arg-type]
                loss = loss + at
                at_sum += at.item()
            if adversary is not None:
                h = maps["gru"].transpose(1, 2)
                spk_y = torch.tensor([spk_index[sampler.data[i].rec.speaker_id] for i in idx])
                adv = F.cross_entropy(adversary(h, mask), spk_y)
                loss = loss + cfg.adversary_weight * adv
                adv_sum += adv.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            ce_sum += ce.item()
        val_metrics = evaluate_frames(net, val, weights, speaker_net, cfg.at_layers)
        row = {
            "epoch": epoch + 1,
            "train_ce": ce_sum / steps,
            "train_at": at_sum / steps if cfg.beta > 0 else None,
            "train_adv": adv_sum / steps if adversary else None,
            **val_metrics,
        }
        history.append(row)
        log(
            f"[turn beta={cfg.beta}{' grl' if adversary else ''}] epoch {epoch + 1}/{cfg.epochs} "
            f"ce={row['train_ce']:.3f}"
            + (f" at={row['train_at']:.3f}" if row["train_at"] is not None else "")
            + f" val_ce={val_metrics['val_ce']:.3f} val_pause_bacc={val_metrics['val_pause_bacc']:.3f}"
            + (
                f" val_at_sim={val_metrics['val_at_similarity']:.3f}"
                if val_metrics.get("val_at_similarity") is not None
                else ""
            )  # fmt: skip
        )
        if val_metrics["val_ce"] < best_val:
            best_val, best_epoch = val_metrics["val_ce"], epoch + 1
            best_state = copy.deepcopy(net.state_dict())
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    report = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_ce": best_val,
        "class_weights": weights.tolist(),
        "steps_per_epoch": steps,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "selection": "min val_seen weighted task CE (AT term excluded)",
    }
    return net, report


@torch.no_grad()
def evaluate_frames(
    net: TurnNet,
    data: Sequence[PreparedRecording],
    weights: torch.Tensor,
    speaker_net: SpeakerNet | None = None,
    at_layers: Sequence[str] = ("conv3",),
) -> dict:
    """Full-recording (streaming-equivalent) frame metrics on held-out data."""
    net.eval()
    ce_num = ce_den = 0.0
    hold_ok = hold_n = shift_ok = shift_n = 0
    sims: list[float] = []
    for d in data:
        x = torch.from_numpy(d.features[None])
        y = torch.from_numpy(d.labels[None])
        logits, maps = net(x)
        flat_y = y.reshape(-1)
        valid = flat_y != IGNORE
        if valid.any():
            ce = F.cross_entropy(logits.reshape(-1, 3)[valid], flat_y[valid], weight=weights,
                                 reduction="sum")  # fmt: skip
            ce_num += float(ce)
            ce_den += float(weights[flat_y[valid]].sum())
        p_end = TurnNet.p_turn_end(logits)[0].numpy()
        lab = d.labels
        hold, shift = lab == HOLD, lab == SHIFT
        hold_ok += int(np.sum(p_end[hold] < 0.5))
        hold_n += int(hold.sum())
        shift_ok += int(np.sum(p_end[shift] >= 0.5))
        shift_n += int(shift.sum())
        if speaker_net is not None:
            _, pre = speaker_net.encoder(x)
            for name in at_layers:
                sims.append(
                    float(gram_similarity(gram_matrix(pre[name]), gram_matrix(maps[name])).mean())
                )
    net.train()
    hold_acc = hold_ok / hold_n if hold_n else float("nan")
    shift_acc = shift_ok / shift_n if shift_n else float("nan")
    return {
        "val_ce": ce_num / max(ce_den, 1e-9),
        "val_hold_acc": hold_acc,
        "val_shift_acc": shift_acc,
        "val_pause_bacc": float(np.nanmean([hold_acc, shift_acc])),
        "val_at_similarity": float(np.mean(sims)) if sims else None,
    }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _plain(obj: object) -> object:
    """Round-trip through JSON so checkpoints hold only types ``weights_only`` loading accepts."""

    def default(o: object) -> object:
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    return json.loads(json.dumps(obj, default=default))


def save_speaker_checkpoint(
    path: Path, net: SpeakerNet, feature_cfg: FeatureConfig, meta: dict
) -> str:
    payload = {
        "format": SPEAKER_FORMAT,
        "metadata": _plain({**meta, "n_speakers": net.n_speakers}),
        "model_config": net.cfg.to_dict(),
        "feature_config": feature_cfg.to_dict(),
        "state_dict": net.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _hash({"meta": payload["metadata"], "model": payload["model_config"]})


def load_speaker_checkpoint(path: Path) -> tuple[SpeakerNet, dict, FeatureConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != SPEAKER_FORMAT:
        raise ValueError(f"{path} is not a speaker-net checkpoint")
    net = SpeakerNet(ModelConfig.from_dict(ck["model_config"]), int(ck["metadata"]["n_speakers"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck["metadata"], FeatureConfig.from_dict(ck["feature_config"])


def save_turn_checkpoint(
    path: Path,
    net: TurnNet,
    mode: str,
    feature_cfg: FeatureConfig,
    train_cfg: TrainConfig,
    *,
    dataset_id: str,
    split_id: str,
    synthetic: bool,
    extra: dict | None = None,
) -> dict:
    ident = {
        "mode": mode,
        "train": train_cfg.to_dict(),
        "model": net.cfg.to_dict(),
        "features": feature_cfg.to_dict(),
        "dataset_id": dataset_id,
        "split_id": split_id,
        "extra": {k: v for k, v in (extra or {}).items() if k == "speaker_checkpoint_hash"},
    }
    version = f"turn-detector-{VARIANT_PREFIX.get(mode, 'x')}-{MODEL_VERSION_BASE}+{_hash(ident)}"
    meta = {
        "model_version": version,
        "mode": mode,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "split_id": split_id,
        "synthetic": synthetic,
        "beta": train_cfg.beta,
        "at_layers": list(train_cfg.at_layers),
        "at_similarity": train_cfg.at_similarity,
        "adversary": train_cfg.adversary,
        "train_config": train_cfg.to_dict(),
        "torch_version": str(torch.__version__),
        **(extra or {}),
    }
    meta = _plain(meta)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "metadata": meta,
        "model_config": net.cfg.to_dict(),
        "feature_config": feature_cfg.to_dict(),
        "state_dict": net.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return meta


def load_turn_checkpoint(path: Path) -> tuple[TurnNet, dict, FeatureConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a turn-detector checkpoint ({ck.get('format')!r})")
    net = TurnNet(ModelConfig.from_dict(ck["model_config"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck["metadata"], FeatureConfig.from_dict(ck["feature_config"])
