"""Characterize model errors without needing a KC-135 subject-matter expert.

Why this exists
---------------
GLIDEPATH Phase 1 originally called for hand-labeling ~100 production records.
That assumed a maintainer was available to do the labeling — reading a write-up
and independently naming the correct WUC is expert judgment, and guessing at it
would manufacture an authoritative-looking number backed by nothing. That is
precisely how "0.903" survived three months in CLAUDE.md while the deployed
model was actually serving 0.7557.

So this splits Phase 1 into what can be computed and what genuinely needs an
expert:

  COMPUTED HERE (no expertise required)
    - exact / subsystem / system agreement, using the WUC code hierarchy
    - the "near miss" rate: wrong WUC but right system, e.g. 45175 vs 45176
      (left vs right hydraulic aux pump) — a very different error from
      predicting a fuselage code for an engine write-up
    - the confusion pairs that account for most disagreement

  NEEDS AN EXPERT (written to adjudication_worksheet.csv)
    - high-confidence disagreements only. When the model says 99% and the QC
      label says otherwise, one of them is wrong. ~25 of these are worth more
      than 102 random records, and are a realistic ask of a working maintainer.

WUC structure assumed: first 2 chars = system, first 3 = subsystem. That is
what main_system.json keys on and what train_hierarchical.py used for its
auxiliary heads.

Usage:
    python training/error_analysis.py [--input app_data_predictions.csv]
                                      [--adjudicate 25]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="app_data_predictions.csv",
                    help="output of batch_predict.py")
    ap.add_argument("--truth-col", default="Corrected WUC")
    ap.add_argument("--adjudicate", type=int, default=25,
                    help="how many high-confidence disagreements to write out")
    ap.add_argument("--min-confidence", type=float, default=90.0,
                    help="confidence floor for the adjudication list")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        print(f"ERROR: {path} not found — run batch_predict.py first", file=sys.stderr)
        return 1

    df = pd.read_csv(path, low_memory=False)
    for col in (args.truth_col, "pred_wuc_1", "confidence_1"):
        if col not in df.columns:
            print(f"ERROR: missing column '{col}'", file=sys.stderr)
            return 1

    df = df[df[args.truth_col].notna()].copy()
    truth = df[args.truth_col].astype(str).str.strip().str.upper()
    pred = df["pred_wuc_1"].astype(str).str.strip().str.upper()

    exact = truth == pred
    subsystem = truth.str[:3] == pred.str[:3]
    system = truth.str[:2] == pred.str[:2]

    n = len(df)
    print("=" * 66)
    print(f"ERROR CHARACTER — {n:,} records from {path.name}")
    print("=" * 66)
    print(f"  exact WUC match          {exact.mean():.4f}")
    print(f"  same subsystem (3 char)  {subsystem.mean():.4f}")
    print(f"  same system (2 char)     {system.mean():.4f}")

    wrong = ~exact
    if wrong.any():
        near = (subsystem & wrong).sum()
        same_sys = (system & wrong).sum()
        far = (~system & wrong).sum()
        w = int(wrong.sum())
        print(f"\n  Of {w:,} disagreements:")
        print(f"    {near:>6,} ({100.0*near/w:5.1f}%) same subsystem — near miss, "
              f"e.g. left vs right of the same component")
        print(f"    {same_sys - near:>6,} ({100.0*(same_sys-near)/w:5.1f}%) same system, "
              f"different subsystem")
        print(f"    {far:>6,} ({100.0*far/w:5.1f}%) different system entirely — "
              f"the genuinely bad errors")
        print("\n  A high near-miss rate means the model understands the write-up "
              "and is\n  splitting hairs the label set draws finely. A high "
              "different-system rate\n  means it is misreading the text.")

    # ---- Confusion pairs --------------------------------------------------
    pairs = Counter(zip(truth[wrong], pred[wrong]))
    if pairs:
        print("\n" + "=" * 66)
        print("TOP CONFUSION PAIRS (truth -> predicted)")
        print("=" * 66)
        print(f"  {'truth':<8} {'predicted':<10} {'n':>6}  {'same sys?':<9}")
        for (t, p), count in pairs.most_common(15):
            print(f"  {t:<8} {p:<10} {count:>6}  {'yes' if t[:2] == p[:2] else 'NO':<9}")
        print("\n  Repeated pairs are relabeling candidates, not model bugs — if two "
              "codes\n  are used interchangeably in the source data, no model can "
              "separate them.")

    # ---- Adjudication worksheet ------------------------------------------
    conf = pd.to_numeric(df["confidence_1"], errors="coerce")
    candidates = df[wrong & (conf >= args.min_confidence)].copy()
    candidates["_conf"] = conf[wrong & (conf >= args.min_confidence)]
    candidates = candidates.sort_values("_conf", ascending=False)

    print("\n" + "=" * 66)
    print("FOR EXPERT ADJUDICATION")
    print("=" * 66)
    print(f"  {len(candidates):,} disagreements at >={args.min_confidence:.0f}% confidence.")
    print("  One side is wrong in each. Either the model erred, or the QC")
    print("  pipeline mislabeled the record — and if it is the latter, every")
    print("  accuracy figure in this repo is UNDERSTATED.")

    if len(candidates):
        take = candidates.head(args.adjudicate)
        cols = [c for c in ["Discrepancy", "Corrective Action", "WCE Narrative",
                            "How Mal", "Action Taken", "model_input"]
                if c in take.columns]
        sheet = take[cols].copy()
        sheet["model_says"] = take["pred_wuc_1"].values
        if "pred_definition_1" in take.columns:
            sheet["model_says_means"] = take["pred_definition_1"].values
        sheet["model_confidence"] = take["_conf"].round(1).values
        sheet["existing_label"] = take[args.truth_col].values
        sheet["WHICH_IS_CORRECT"] = ""   # expert fills: MODEL / LABEL / NEITHER
        sheet["correct_wuc_if_neither"] = ""
        sheet["expert_notes"] = ""

        out = REPO_ROOT / "adjudication_worksheet.csv"
        sheet.to_csv(out, index=False)
        print(f"\n  Wrote {out.name}: {len(sheet)} records, highest confidence first.")
        print("  Ask a maintainer to fill WHICH_IS_CORRECT with MODEL, LABEL, or")
        print("  NEITHER. That is a ~30 minute task, not a multi-session one.")

    print("\n  NOTE: nothing here is a hand-verified accuracy figure. This")
    print("  characterizes errors against the existing labels. Only the")
    print("  adjudication worksheet can tell you whether those labels are right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
