"""Speaker-leakage probe (goal.md §7, "could have").

A linear (multinomial logistic) classifier is trained on frozen, utterance-pooled
representations (mean ⊕ std over each IPU's frames) to predict the speaker. Utterances of
each speaker are split within-speaker into probe-train / probe-test, so the probe measures
how linearly decodable speaker identity is from the representation.

Reference points reported alongside the turn models:

* ``random_init`` — the same architecture with random weights (random features of spectra
  already carry a lot of voice information; leakage should be judged against this);
* ``speaker_net`` — the frozen orthogonal-task network (an approximate upper reference).

Caveats: lower probe accuracy suggests less *linearly* decodable speaker information at the
probed layer; it does not prove speaker information is absent (non-linear probes, other
layers), and it says nothing on its own about turn-detection quality. Interpret together
with the task metrics (goal.md §7).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from turnsense.research.data import PreparedRecording
from turnsense.research.model import Encoder


@torch.no_grad()
def utterance_embeddings(
    encoder: Encoder, data: Sequence[PreparedRecording], layer: str, hop_ms: float = 10.0
) -> tuple[np.ndarray, list[str]]:
    encoder.eval()
    xs: list[np.ndarray] = []
    ys: list[str] = []
    for d in data:
        _, maps = encoder(torch.from_numpy(d.features[None]))
        fmap = maps[layer][0]  # [C, T]
        for turn in d.rec.conversation_turns:
            for a, b in turn.ipus or [(turn.start_ms, turn.end_ms)]:
                i0, i1 = int(a / hop_ms), min(int(b / hop_ms), fmap.shape[1])
                if i1 - i0 < 10:
                    continue
                seg = fmap[:, i0:i1]
                xs.append(torch.cat([seg.mean(1), seg.std(1)]).numpy())
                ys.append(d.rec.speaker_id)
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(xs).astype(np.float32), ys


def linear_probe(
    x: np.ndarray,
    y: Sequence[str],
    seed: int = 0,
    test_frac: float = 0.3,
    weight_decay: float = 1e-3,
) -> dict:
    """Within-speaker split; standardised features; L2-regularised logistic regression."""
    rng = np.random.default_rng(seed)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(y):
        by_spk[s].append(i)
    classes = sorted(s for s, idx in by_spk.items() if len(idx) >= 3)
    if len(classes) < 2:
        return {
            "accuracy": None,
            "chance": None,
            "n_classes": len(classes),
            "note": "too few utterances",
        }
    cls_index = {s: i for i, s in enumerate(classes)}
    tr, te = [], []
    for s in classes:
        idx = list(by_spk[s])
        rng.shuffle(idx)
        n_te = max(1, int(round(test_frac * len(idx))))
        te += idx[:n_te]
        tr += idx[n_te:]
    xt, xv = x[tr], x[te]
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    xt_t = torch.from_numpy((xt - mu) / sd)
    xv_t = torch.from_numpy((xv - mu) / sd)
    yt = torch.tensor([cls_index[y[i]] for i in tr])
    yv = torch.tensor([cls_index[y[i]] for i in te])
    torch.manual_seed(seed)
    lin = torch.nn.Linear(x.shape[1], len(classes))
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(lin(xt_t), yt) + weight_decay * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        acc = float((lin(xv_t).argmax(1) == yv).float().mean())
        train_acc = float((lin(xt_t).argmax(1) == yt).float().mean())
    counts = np.bincount(yv.numpy(), minlength=len(classes))
    return {
        "accuracy": acc,
        "train_accuracy": train_acc,
        "chance": float(counts.max() / counts.sum()),
        "uniform_chance": 1.0 / len(classes),
        "n_classes": len(classes),
        "n_train": len(tr),
        "n_test": len(te),
    }


def run_probe(
    encoders: dict[str, Encoder],
    data_by_set: dict[str, Sequence[PreparedRecording]],
    layers: Sequence[str],
    seed: int = 0,
) -> dict:
    out: dict[str, dict] = {}
    for set_name, data in data_by_set.items():
        out[set_name] = {}
        for enc_name, enc in encoders.items():
            out[set_name][enc_name] = {}
            for layer in layers:
                x, y = utterance_embeddings(enc, data, layer)
                out[set_name][enc_name][layer] = (
                    linear_probe(x, y, seed) if len(y) else {"accuracy": None}
                )
    return out


def probe_markdown(result: dict) -> str:
    lines = [
        "## Speaker-leakage probe (linear, utterance-pooled; lower = less linearly decodable speaker identity)",
        "",
        "| speaker set | representation | layer | probe accuracy | majority chance | classes | test utterances |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for set_name, encs in result.items():
        for enc_name, layers in encs.items():
            for layer, r in layers.items():
                acc = "—" if r.get("accuracy") is None else f"{100 * r['accuracy']:.1f}%"
                ch = "—" if r.get("chance") is None else f"{100 * r['chance']:.1f}%"
                lines.append(
                    f"| {set_name} | {enc_name} | {layer} | {acc} | {ch} | {r.get('n_classes', '—')} | {r.get('n_test', '—')} |"
                )
    lines += [
        "",
        "Caveat: a linear probe bounds only *linearly* decodable information at one layer; "
        "`random_init` shows how much voice information untrained features already carry.",
        "",
    ]
    return "\n".join(lines)
