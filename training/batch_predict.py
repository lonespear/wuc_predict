"""Batch WUC prediction — CSV in, top-k predictions + confidences out.

Runs entirely on the GPU box. Built for two jobs:

  1. Bulk re-validation of maintenance records.
  2. Generating the hand-labeling worksheet for Phase 1 of GLIDEPATH.md —
     the 100 production records that turn "0.903 on a test set drawn from the
     same QC pipeline as training" into a number worth defending.

Input format
------------
Any CSV with a `Discrepancy` column. `Corrective Action`, `WCE Narrative`,
`How Mal` and `Action Taken` are used when present. Field order, the
" [SEP] " separator and the skip-empty rule mirror `prepare_data.py`'s
TEXT_FIELDS exactly — see build_text() for the one deliberate difference.

Usage
-----
    export WUC_MODEL_PATH=./wuc-model-hier
    python training/batch_predict.py --input app_data.csv --limit 5000

    # Phase 1 labeling worksheet — 34 records per confidence band
    python training/batch_predict.py --input app_data.csv \
        --limit 20000 --worksheet 34
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

# model_loader opens codes.json / main_system.json / the lookup CSV by
# *relative* path, so it only imports cleanly from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

if not os.environ.get("WUC_MODEL_PATH"):
    sys.exit(
        "ERROR: WUC_MODEL_PATH is not set.\n"
        "Without it model_loader falls back to the legacy HF model "
        "'jonday/wuc-model' — a different checkpoint with a different label "
        "space (1727 classes vs 1251). It would load without complaint and "
        "produce confident, wrong predictions.\n"
        "Fix: export WUC_MODEL_PATH=./wuc-model-hier"
    )

from model_loader import (  # noqa: E402
    model, tokenizer, index_to_wuc, wuc_defs, main_system, MAX_LEN,
)

# Must match prepare_data.py TEXT_FIELDS, in this order.
TEXT_FIELDS = ["Discrepancy", "Corrective Action", "WCE Narrative", "How Mal", "Action Taken"]

SEED = 42  # matches prepare_data.py, so samples are reproducible


def build_text(row: pd.Series) -> str:
    """Join the five training fields exactly as prepare_data.py did.

    Two details that matter:

    * `str(v)` coercion is explicit. `model_loader.build_input_text()` guards
      with `isinstance(value, str)`, which is fine for Streamlit (widgets
      always return str) but silently DROPS short code columns like `How Mal`
      and `Action Taken` when pandas parses them as numeric. That would build
      systematically shorter text than training used and depress the accuracy
      number for reasons unrelated to the model.

    * `.upper()` is applied to match `build_input_text()`. Training did not
      uppercase, but the source text is 100% uppercase already (audited
      2026-07-31, n=132,962), so this is a verified no-op on real data and
      protects against anything hand-entered.
    """
    parts = []
    for col in TEXT_FIELDS:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            parts.append(str(v).strip().upper())
    return " [SEP] ".join(parts)


def band(confidence: float) -> str:
    """Confidence bands as shown in the Tab 1 UI."""
    if confidence >= 70:
        return "high (>=70%)"
    if confidence >= 30:
        return "mid (30-70%)"
    return "low (<30%)"


def predict_batches(texts: list[str], batch_size: int, top_k: int,
                    device: torch.device, max_length: int):
    """Batched top-k inference. Returns (wucs, confidences) as nested lists.

    max_length MUST match training (train_hierarchical.py MAX_LEN = 128).
    Omitting it lets the tokenizer fall back to model_max_length (8192 for
    ModernBERT), so long write-ups reach the classifier at sequence lengths
    it never trained on — a silent accuracy loss, not an error."""
    all_wucs, all_confs = [], []
    total = len(texts)

    for start in range(0, total, batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                        padding=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        k = min(top_k, probs.shape[1])
        top_probs, top_idx = torch.topk(probs, k=k, dim=1)

        for row_probs, row_idx in zip(top_probs.tolist(), top_idx.tolist()):
            all_wucs.append([index_to_wuc.get(int(i), "Unknown WUC") for i in row_idx])
            all_confs.append([float(p) * 100.0 for p in row_probs])

        done = min(start + batch_size, total)
        print(f"  {done:,}/{total:,}", end="\r", flush=True)

    print()
    return all_wucs, all_confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="CSV or .parquet with a Discrepancy column")
    ap.add_argument("--text-col", default=None, metavar="COL",
                    help="use an existing pre-built input column instead of "
                         "rebuilding from the five text fields. Use "
                         "--text-col text on data_splits/*.parquet to "
                         "reproduce the training-time eval exactly")
    ap.add_argument("--output", default=None, help="default: <input>_predictions.csv")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=MAX_LEN,
                    help=f"tokenizer truncation length (default {MAX_LEN}, "
                         f"matching training). Changing this invalidates "
                         f"comparison with the reported model metrics")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows")
    ap.add_argument("--sample", type=int, default=None, metavar="N",
                    help="random sample of N rows (seeded). Prefer this over "
                         "--limit for anything you intend to quote — --limit "
                         "takes the head of the file, which is not a sample")
    ap.add_argument("--exclude-seen", nargs="*", default=None, metavar="PARQUET",
                    help="drop rows the model was trained on, matched on the "
                         "built input text. Bare flag uses "
                         "data_splits/train.parquet + val.parquet")
    ap.add_argument("--truth-col", default="Corrected WUC",
                    help="ground-truth column for scoring; skipped if absent")
    ap.add_argument("--worksheet", type=int, default=0, metavar="N",
                    help="also write a hand-labeling worksheet, N records per "
                         "confidence band (stratified, not random)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        return 1

    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path, low_memory=False)

    if args.text_col:
        if args.text_col not in df.columns:
            print(f"ERROR: --text-col '{args.text_col}' not in {in_path.name}. "
                  f"Found: {list(df.columns)[:12]}...", file=sys.stderr)
            return 1
    elif "Discrepancy" not in df.columns:
        print(f"ERROR: no 'Discrepancy' column in {in_path.name}. "
              f"Found: {list(df.columns)[:10]}...", file=sys.stderr)
        return 1
    present = [c for c in TEXT_FIELDS if c in df.columns]
    absent = [c for c in TEXT_FIELDS if c not in df.columns]
    print(f"Loaded {len(df):,} rows from {in_path.name}")
    print(f"Text fields present: {present}")
    if absent:
        print(f"Text fields absent (skipped, as in training): {absent}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model: {os.environ['WUC_MODEL_PATH']} on {device} "
          f"({len(index_to_wuc):,} classes)")

    # Build input text for EVERY row before subsetting — the training-overlap
    # anti-join below matches on it.
    if args.text_col:
        df["model_input"] = df[args.text_col].astype(str)
        print(f"Using pre-built text from column '{args.text_col}' "
              f"(not rebuilding from the five fields)")
    else:
        df["model_input"] = df.apply(build_text, axis=1)

    if args.exclude_seen is not None:
        split_paths = args.exclude_seen or [
            "data_splits/train.parquet", "data_splits/val.parquet"
        ]
        seen: set[str] = set()
        for sp in split_paths:
            p = Path(sp)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if not p.exists():
                print(f"WARNING: {p} not found — cannot exclude it", file=sys.stderr)
                continue
            split = pd.read_parquet(p, columns=["text"])
            seen.update(split["text"].astype(str))
            print(f"Excluding {len(split):,} rows seen in {p.name}")
        if seen:
            before = len(df)
            df = df[~df["model_input"].isin(seen)]
            print(f"Training overlap removed: {before:,} -> {len(df):,} "
                  f"({before - len(df):,} dropped)")

    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=SEED)
        print(f"Random sample (seed {SEED}): {len(df):,} rows")
    elif args.limit:
        df = df.head(args.limit)
        print(f"NOTE: --limit takes the FIRST {args.limit:,} rows, not a "
              f"sample. Do not quote accuracy from this; use --sample.")
    df = df.reset_index(drop=True)

    if df.empty:
        print("ERROR: no rows left to predict", file=sys.stderr)
        return 1

    texts = df["model_input"].tolist()
    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"WARNING: {empty:,} rows produced empty input text")

    print("Predicting...")
    wucs, confs = predict_batches(texts, args.batch_size, args.top_k, device,
                                  args.max_length)

    out = df.copy()
    for i in range(args.top_k):
        out[f"pred_wuc_{i + 1}"] = [w[i] if i < len(w) else None for w in wucs]
        out[f"confidence_{i + 1}"] = [round(c[i], 2) if i < len(c) else None for c in confs]
    out["pred_definition_1"] = [
        wuc_defs.get(w[0], f"(no dictionary entry for {w[0]})") if w else None for w in wucs
    ]
    out["pred_system_1"] = [
        main_system.get(w[0][:2], "Unknown Main System") if w else None for w in wucs
    ]
    out["confidence_band"] = [band(c[0]) if c else "low (<30%)" for c in confs]

    out_path = Path(args.output) if args.output else in_path.with_name(
        in_path.stem + "_predictions.csv")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.name}: {len(out):,} rows")

    # ---- Scoring, when ground truth is available --------------------------
    truth = args.truth_col
    if truth in out.columns:
        scored = out[out[truth].notna()].copy()
        if len(scored):
            t = scored[truth].astype(str).str.strip().str.upper()
            p1 = scored["pred_wuc_1"].astype(str).str.strip().str.upper()
            scored["top1_correct"] = t == p1
            topk_cols = [f"pred_wuc_{i + 1}" for i in range(args.top_k)
                         if f"pred_wuc_{i + 1}" in scored.columns]
            scored["topk_correct"] = [
                any(str(r[c]).strip().upper() == tv for c in topk_cols)
                for (_, r), tv in zip(scored.iterrows(), t)
            ]

            # A record whose true WUC is not among the model's classes cannot
            # be got right — prepare_data.py's MIN_PER_CLASS filter drops rare
            # WUCs from the label map entirely. Mixing those into the headline
            # accuracy conflates two different problems: how good the model is,
            # and what share of reality it can express.
            label_space = {str(v).strip().upper() for v in index_to_wuc.values()}
            scored["label_in_model_space"] = t.isin(label_space)
            answerable = scored[scored["label_in_model_space"]]
            impossible = int((~scored["label_in_model_space"]).sum())

            print(f"\nScored against '{truth}' ({len(scored):,} labeled rows):")
            print(f"  top-1 accuracy (all rows):   {scored['top1_correct'].mean():.4f}")
            print(f"  top-{args.top_k} accuracy (all rows):   {scored['topk_correct'].mean():.4f}")
            if impossible:
                print(f"\n  Label coverage: {len(answerable):,} of {len(scored):,} "
                      f"({100.0 * len(answerable) / len(scored):.1f}%) have a true "
                      f"WUC the model can emit.")
                print(f"  {impossible:,} rows ({100.0 * impossible / len(scored):.1f}%) "
                      f"are unanswerable — their WUC was below MIN_PER_CLASS and "
                      f"never entered the label map.")
                if len(answerable):
                    print(f"\n  On answerable rows only:")
                    print(f"    top-1: {answerable['top1_correct'].mean():.4f}")
                    print(f"    top-{args.top_k}: {answerable['topk_correct'].mean():.4f}")
                print("\n  Report BOTH: accuracy is model quality, coverage is a "
                      "data/label-map decision. They have different fixes.")

            print("\n  Calibration by confidence band (answerable rows):")
            print(f"  {'band':<14} {'n':>8} {'top-1 acc':>10}")
            basis = answerable if len(answerable) else scored
            for b in ("high (>=70%)", "mid (30-70%)", "low (<30%)"):
                sub = basis[basis["confidence_band"] == b]
                if len(sub):
                    print(f"  {b:<14} {len(sub):>8,} {sub['top1_correct'].mean():>10.4f}")
            print("\n  NOTE: this scores against the same QC pipeline that "
                  "produced training labels. It is not the hand-checked "
                  "number Phase 1 is after — use --worksheet for that.")
    else:
        print(f"\nNo '{truth}' column — skipping scoring.")

    # ---- Phase 1 labeling worksheet ---------------------------------------
    if args.worksheet:
        keep = [c for c in present] + [
            f"pred_wuc_{i + 1}" for i in range(args.top_k)
        ] + ["confidence_1", "pred_definition_1", "confidence_band", "model_input"]
        if args.truth_col in out.columns:
            keep.append(args.truth_col)  # the QC-pipeline label, to compare against
        keep = [c for c in keep if c in out.columns]

        chunks = []
        for b in ("high (>=70%)", "mid (30-70%)", "low (<30%)"):
            sub = out[out["confidence_band"] == b]
            if len(sub):
                chunks.append(sub.sample(min(args.worksheet, len(sub)), random_state=SEED))
            else:
                print(f"WARNING: no rows in band {b}")
        sheet = pd.concat(chunks, ignore_index=True)[keep]

        # Blank columns for you to fill in by hand.
        sheet["true_wuc"] = ""
        sheet["top1_correct"] = ""
        sheet["top3_correct"] = ""
        sheet["notes"] = ""

        ws_path = REPO_ROOT / "labeling_worksheet.csv"
        sheet.to_csv(ws_path, index=False)
        print(f"\nWrote {ws_path.name}: {len(sheet):,} records, "
              f"{args.worksheet} per band.")
        print("  Equal-per-band deliberately oversamples the 30-70% range — "
              "random sampling wastes the effort on easy cases.")
        print("  Fill true_wuc / top1_correct / top3_correct by hand, then "
              "that file IS the Phase 1 deliverable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
