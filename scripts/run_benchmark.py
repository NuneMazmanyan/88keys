"""Run transcription models on MAESTRO-Wild and score them.

For every model x condition x piece: transcribe (cached), evaluate against the labels,
and write
  outputs/benchmark/results.csv     one row per model, condition, piece (all metrics)
  outputs/benchmark/summary.md      mean note F1 / note-with-offset F1 per model and condition

`oracle` returns the ground truth itself: it must score 1.0 and checks the harness.

  python scripts/run_benchmark.py --models oracle reference
  python scripts/run_benchmark.py --models reference --conditions clean noise_10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pianoscribe.data.audio import SAMPLE_RATE, load_audio
from pianoscribe.data.notes import Performance, extend_offsets_by_pedal
from pianoscribe.eval.metrics import evaluate
from pianoscribe.transcribers import get_transcriber


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wild", type=Path, default=Path("data/processed/maestro_wild"))
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    ap.add_argument("--models", nargs="+", default=["oracle", "reference"])
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    manifest = pd.read_csv(args.wild / "manifest.csv")
    conditions = args.conditions or pd.read_csv(args.wild / "conditions.csv")["condition"].tolist()
    rows = []

    for model_name in args.models:
        model = None if model_name == "oracle" else get_transcriber(model_name, args.device)
        jobs = [(c, pid) for c in conditions for pid in manifest["id"]]
        for cond, pid in tqdm(jobs, desc=model_name):
            ref = Performance.load(args.wild / "labels" / f"{pid}.npz")
            cache = args.out / "cache" / model_name / cond / f"{pid}.npz"
            seconds = float("nan")
            if model is None:  # oracle: ground truth with the same pedal convention
                est = Performance(notes=extend_offsets_by_pedal(ref), pedal=ref.pedal)
            elif cache.exists():
                est = Performance.load(cache)
            else:
                audio = load_audio(args.wild / cond / f"{pid}.flac")
                t0 = time.perf_counter()
                est = model.transcribe(audio)
                seconds = time.perf_counter() - t0
                cache.parent.mkdir(parents=True, exist_ok=True)
                est.save(cache)
                pd.Series({"seconds": seconds, "audio_seconds": len(audio) / SAMPLE_RATE}).to_json(
                    cache.with_suffix(".json")
                )
            if cache.with_suffix(".json").exists():
                seconds = pd.read_json(cache.with_suffix(".json"), typ="series")["seconds"]
            rows.append(
                {"model": model_name, "condition": cond, "id": pid,
                 "seconds": seconds, **evaluate(ref, est)}
            )  # fmt: skip

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "results.csv", index=False)

    order = {c: i for i, c in enumerate(conditions)}
    lines = ["# MAESTRO-Wild benchmark", "", f"{manifest.shape[0]} pieces, "
             f"{manifest['duration'].sum() / 60:.1f} min of audio per condition.", ""]  # fmt: skip
    for metric, title in (("note_f1", "Note F1 (onset)"), ("note_off_f1", "Note F1 with offset")):
        table = df.pivot_table(index="condition", columns="model", values=metric, aggfunc="mean")
        table = table.loc[sorted(table.index, key=order.get)]
        lines += [f"## {title}", "", (100 * table).round(1).to_markdown(), ""]
    timing = df.dropna(subset=["seconds"])
    if not timing.empty:
        rtf = timing.groupby("model")["seconds"].sum() / (
            manifest["duration"].sum() * len(conditions)
        )
        lines += ["## Speed (real-time factor, lower is faster)", "",
                  rtf.round(3).to_frame("RTF").to_markdown(), ""]  # fmt: skip
    (args.out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
