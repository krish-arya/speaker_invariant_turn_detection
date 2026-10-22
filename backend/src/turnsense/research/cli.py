"""``turnsense-research`` command line.

    turnsense-research pipeline --synthetic --quick          # end-to-end smoke test
    turnsense-research synth --out runs/syn --quick
    turnsense-research split --manifest M --out split.json --unseen-frac 0.3 --val-frac 0.2
    turnsense-research train-speaker --manifest M --split S --out speaker_net.pt
    turnsense-research train --manifest M --split S --mode anti_transfer --speaker-ckpt P --out at.pt
    turnsense-research evaluate --manifest M --split S --checkpoint learned_control=lc.pt \
        --checkpoint anti_transfer=at.pt --out report_dir
    turnsense-research probe --manifest M --split S --speaker-ckpt P --checkpoint ... --out dir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from turnsense.config import BACKEND_ROOT
from turnsense.events import DetectorMode, PolicyConfig
from turnsense.research.data import (
    PreparedRecording,
    Recording,
    Split,
    build_splits,
    load_manifest,
    prepare,
)
from turnsense.research.evaluate import SYNTHETIC_BANNER, default_factories, evaluate, write_report
from turnsense.research.features import FeatureConfig
from turnsense.research.model import ModelConfig, TurnNet
from turnsense.research.probe import probe_markdown, run_probe
from turnsense.research.synthetic import QUICK, SyntheticConfig, generate_corpus
from turnsense.research.train import (
    SpeakerTrainConfig,
    TrainConfig,
    feature_stats,
    load_speaker_checkpoint,
    load_turn_checkpoint,
    save_speaker_checkpoint,
    save_turn_checkpoint,
    train_speaker_net,
    train_turn_net,
)


def _dataset_id(records: list[Recording], manifest: Path) -> str:
    ids = {str(r.meta.get("dataset_id")) for r in records if r.meta.get("dataset_id")}
    if len(ids) == 1:
        return ids.pop()
    return "manifest-" + hashlib.sha256(manifest.read_bytes()).hexdigest()[:10]


def _split_data(
    prepared: list[PreparedRecording], split: Split
) -> dict[str, list[PreparedRecording]]:
    out: dict[str, list[PreparedRecording]] = {"train": [], "val_seen": [], "test_unseen": []}
    for p in prepared:
        name = split.assignments.get(p.rec.recording_id)
        if name in out:
            out[name].append(p)
    return out


def _load(manifest: Path, split_path: Path, cache: Path | None):
    records = load_manifest(manifest)
    split = Split.load(split_path)
    feature_cfg = FeatureConfig()
    keep = [r for r in records if r.recording_id in split.assignments]
    prepared = prepare(keep, feature_cfg, cache_dir=cache or manifest.parent / ".cache")
    return records, split, feature_cfg, _split_data(prepared, split)


def _parse_ckpts(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint expects name=path, got {item!r}")
        out[name] = Path(path)
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> None:
    cfg = replace(QUICK if args.quick else SyntheticConfig(), seed=args.seed)
    manifest, unseen, info = generate_corpus(cfg, Path(args.out))
    (Path(args.out) / "unseen_speakers.json").write_text(json.dumps(unseen), encoding="utf-8")
    print(
        json.dumps(
            {"manifest": str(manifest), "unseen_speakers": len(unseen), **info["confound_summary"]},
            indent=2,
        )
    )


def cmd_split(args: argparse.Namespace) -> None:
    records = load_manifest(Path(args.manifest))
    unseen = None
    if args.unseen_speakers:
        p = Path(args.unseen_speakers)
        unseen = json.loads(p.read_text()) if p.is_file() else args.unseen_speakers.split(",")
    split = build_splits(records, args.unseen_frac, args.val_frac, args.seed, unseen)
    split.save(Path(args.out))
    print(json.dumps({"split_id": split.split_id, "stats": split.stats}, indent=2))


def cmd_train_speaker(args: argparse.Namespace) -> None:
    records, split, feature_cfg, data = _load(Path(args.manifest), Path(args.split), None)
    model_cfg = ModelConfig(input_dim=feature_cfg.dim)
    cfg = SpeakerTrainConfig(seed=args.seed, epochs=args.epochs)
    net, report = train_speaker_net(data["train"], data["val_seen"], model_cfg, feature_cfg, cfg)
    save_speaker_checkpoint(Path(args.out), net, feature_cfg, {
        "dataset_id": _dataset_id(records, Path(args.manifest)), "split_id": split.split_id,
        "train_config": cfg.to_dict(), "report": report})  # fmt: skip


def cmd_train(args: argparse.Namespace) -> None:
    records, split, feature_cfg, data = _load(Path(args.manifest), Path(args.split), None)
    model_cfg = ModelConfig(input_dim=feature_cfg.dim)
    spk = None
    if args.speaker_ckpt:
        spk, _, _ = load_speaker_checkpoint(Path(args.speaker_ckpt))
    beta = args.beta if args.mode == DetectorMode.ANTI_TRANSFER.value else 0.0
    cfg = TrainConfig(
        seed=args.seed, epochs=args.epochs, beta=beta, at_layers=tuple(args.at_layer),
        adversary="grl" if args.mode == "speaker_adversarial" else None,
    )  # fmt: skip
    net, report = train_turn_net(data["train"], data["val_seen"], model_cfg, feature_cfg, cfg, spk)
    meta = save_turn_checkpoint(
        Path(args.out), net, args.mode, feature_cfg, cfg,
        dataset_id=_dataset_id(records, Path(args.manifest)), split_id=split.split_id,
        synthetic=all(r.synthetic for r in records), extra={"train_report": report},
    )  # fmt: skip
    print(json.dumps({"model_version": meta["model_version"], "best_epoch": report["best_epoch"]}))


def cmd_evaluate(args: argparse.Namespace) -> None:
    records = load_manifest(Path(args.manifest))
    split = Split.load(Path(args.split))
    report = evaluate(
        records, split, default_factories(_parse_ckpts(args.checkpoint)), PolicyConfig()
    )
    write_report(report, Path(args.out), synthetic=all(r.synthetic for r in records))
    print((Path(args.out) / "evaluation.md").read_text(encoding="utf-8"))


def cmd_probe(args: argparse.Namespace) -> None:
    records, split, feature_cfg, data = _load(Path(args.manifest), Path(args.split), None)
    encoders = {}
    if args.speaker_ckpt:
        spk, _, _ = load_speaker_checkpoint(Path(args.speaker_ckpt))
        encoders["speaker_net"] = spk.encoder
    for name, path in _parse_ckpts(args.checkpoint).items():
        net, _, _ = load_turn_checkpoint(path)
        encoders[name] = net.encoder
    result = run_probe(encoders, {"val_seen": data["val_seen"], "test_unseen": data["test_unseen"]},
                       args.layer, seed=args.seed)  # fmt: skip
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(probe_markdown(result))


def cmd_pipeline(args: argparse.Namespace) -> None:
    if not args.synthetic:
        raise SystemExit("pipeline currently runs the synthetic smoke test only; use the "
                         "individual subcommands for a real manifest")  # fmt: skip
    t_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(args.out)
        if args.out
        else BACKEND_ROOT / "runs" / f"{stamp}-synthetic{'-quick' if args.quick else ''}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[pipeline] {SYNTHETIC_BANNER}")
    log(f"[pipeline] run dir: {run_dir}")
    syn_cfg = replace(QUICK if args.quick else SyntheticConfig(), seed=args.seed)
    t = time.perf_counter()
    manifest, unseen, info = generate_corpus(syn_cfg, run_dir / "data")
    log(f"[synth] {info['n_recordings']} recordings in {time.perf_counter() - t:.1f}s; "
        f"confound: {json.dumps(info['confound_summary'])}")  # fmt: skip

    records = load_manifest(manifest)
    split = build_splits(records, val_frac=0.25, seed=args.seed, unseen_speakers=unseen)
    split.save(run_dir / "split.json")
    log(f"[split] {split.split_id}: {json.dumps(split.stats)}")

    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig(input_dim=feature_cfg.dim)
    t = time.perf_counter()
    prepared = prepare(records, feature_cfg, cache_dir=run_dir / "cache")
    data = _split_data(prepared, split)
    log(
        f"[features] {sum(p.features.shape[0] for p in prepared)} frames in {time.perf_counter() - t:.1f}s"
    )

    epochs = args.epochs or (8 if args.quick else 20)
    spk_cfg = SpeakerTrainConfig(seed=args.seed, epochs=max(3, epochs // 2))
    spk_net, spk_report = train_speaker_net(
        data["train"], data["val_seen"], model_cfg, feature_cfg, spk_cfg, log
    )
    ckpt_dir = run_dir / "checkpoints"
    spk_hash = save_speaker_checkpoint(ckpt_dir / "speaker_net.pt", spk_net, feature_cfg, {
        "dataset_id": syn_cfg.dataset_id, "split_id": split.split_id, "synthetic": True,
        "train_config": spk_cfg.to_dict(), "report": spk_report})  # fmt: skip

    base_cfg = TrainConfig(seed=args.seed, epochs=epochs, at_layers=tuple(args.at_layer))
    variants: dict[str, TrainConfig] = {
        DetectorMode.LEARNED_CONTROL.value: replace(base_cfg, beta=0.0),
        DetectorMode.ANTI_TRANSFER.value: replace(base_cfg, beta=args.beta),
    }
    if args.grl:
        variants["speaker_adversarial"] = replace(base_cfg, beta=0.0, adversary="grl")
    nets: dict[str, TurnNet] = {}
    train_reports: dict[str, dict] = {}
    ckpt_paths: dict[str, Path] = {}
    for mode, cfg in variants.items():
        # the speaker net is passed to every variant for the val Gram-similarity diagnostic;
        # it only enters the loss when beta > 0
        net, rep = train_turn_net(
            data["train"], data["val_seen"], model_cfg, feature_cfg, cfg, spk_net, log
        )
        nets[mode], train_reports[mode] = net, rep
        path = ckpt_dir / f"{mode}.pt"
        meta = save_turn_checkpoint(
            path, net, mode, feature_cfg, cfg, dataset_id=syn_cfg.dataset_id, split_id=split.split_id,
            synthetic=True, extra={"speaker_checkpoint_hash": spk_hash, "best_epoch": rep["best_epoch"],
                                   "run_dir": run_dir.name},
        )  # fmt: skip
        ckpt_paths[mode] = path
        log(
            f"[train] {mode}: {meta['model_version']} (best epoch {rep['best_epoch']}, {rep['train_seconds']}s)"
        )

    report = evaluate(records, split, default_factories(ckpt_paths), PolicyConfig(), log=log)

    torch.manual_seed(args.seed)
    rand = TurnNet(model_cfg)
    rand.encoder.set_normalisation(*feature_stats(data["train"]))
    encoders = {"random_init": rand.encoder, "speaker_net": spk_net.encoder}
    encoders.update({m: n.encoder for m, n in nets.items()})
    probe = run_probe(encoders, {"val_seen": data["val_seen"], "test_unseen": data["test_unseen"]},
                      layers=[*args.at_layer, "gru"], seed=args.seed)  # fmt: skip

    training_md = [
        "## Training summary",
        "",
        f"- Speaker net (orthogonal task, training speakers only): {spk_report['n_speakers']} speakers, "
        f"final val_seen accuracy {spk_report['history'][-1]['val_seen_acc']:.3f} (chance {spk_report['chance']:.3f})",
        "",
        "| variant | β | AT layer(s) | best epoch | val_seen task CE | val_seen pause bal. acc | val_seen Gram cos² vs speaker net |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for mode, rep in train_reports.items():
        best = rep["history"][rep["best_epoch"] - 1]
        sim = best.get("val_at_similarity")
        training_md.append(
            f"| {mode} | {variants[mode].beta} | {', '.join(variants[mode].at_layers)} | {rep['best_epoch']} | "
            f"{best['val_ce']:.3f} | {best['val_pause_bacc']:.3f} | {'—' if sim is None else f'{sim:.3f}'} |"
        )
    training_md.append("")
    write_report(
        report, run_dir, synthetic=True, extra_sections=[*training_md, probe_markdown(probe)]
    )
    (run_dir / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    (run_dir / "pipeline.json").write_text(json.dumps({
        "SYNTHETIC": True,
        "synthetic_config": syn_cfg.to_dict(),
        "dataset_info": info,
        "split_id": split.split_id,
        "feature_config": feature_cfg.to_dict(),
        "model_config": model_cfg.to_dict(),
        "speaker_training": spk_report,
        "turn_training": train_reports,
        "train_configs": {m: c.to_dict() for m, c in variants.items()},
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }, indent=2, default=str), encoding="utf-8")  # fmt: skip
    (run_dir / "pipeline.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if args.install:
        dest = BACKEND_ROOT / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        for mode in (DetectorMode.LEARNED_CONTROL.value, DetectorMode.ANTI_TRANSFER.value):
            shutil.copy2(ckpt_paths[mode], dest / f"{mode}.pt")
        log(f"[pipeline] installed checkpoints into {dest} (SYNTHETIC-trained)")
    log(f"[pipeline] done in {time.perf_counter() - t_start:.0f}s -> {run_dir / 'evaluation.md'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turnsense-research", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a SYNTHETIC corpus + manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--quick", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("split", help="build a speaker-disjoint split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--unseen-frac", type=float, default=0.3)
    s.add_argument("--val-frac", type=float, default=0.2)
    s.add_argument("--unseen-speakers", help="JSON file or comma list of speaker ids to hold out")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("train-speaker", help="pre-train the orthogonal-task speaker network")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--epochs", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train_speaker)

    s = sub.add_parser("train", help="train a turn model")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--mode", required=True,
                   choices=[DetectorMode.LEARNED_CONTROL.value, DetectorMode.ANTI_TRANSFER.value, "speaker_adversarial"])  # fmt: skip
    s.add_argument("--speaker-ckpt")
    s.add_argument("--beta", type=float, default=1.0)
    s.add_argument("--at-layer", action="append", default=None)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="streaming evaluation on val_seen and test_unseen")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--checkpoint", action="append", default=[], help="name=path (repeatable)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("probe", help="speaker-leakage probe")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--speaker-ckpt")
    s.add_argument("--checkpoint", action="append", default=[])
    s.add_argument("--layer", action="append", default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("pipeline", help="synth → split → train → evaluate → probe")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--quick", action="store_true")
    s.add_argument("--out")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--epochs", type=int)
    s.add_argument("--beta", type=float, default=1.0)
    s.add_argument("--at-layer", action="append", default=None)
    s.add_argument("--grl", action="store_true", help="also train the gradient-reversal ablation")
    s.add_argument("--no-install", dest="install", action="store_false",
                   help="do not copy checkpoints into backend/checkpoints/")  # fmt: skip
    s.set_defaults(func=cmd_pipeline)
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    for attr in ("at_layer", "layer"):
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, ["conv3"] if attr == "at_layer" else ["conv3", "gru"])
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
