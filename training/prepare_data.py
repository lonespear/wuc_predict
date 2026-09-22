"""Turn the combined KC-135 corpus into train/val/test splits for WUC prediction.

Input:
    data/combined_*.csv — built by training/build_corpus.py (one row per job,
    every extract merged). Run that first.

Outputs (in ./data_splits/):
    train.parquet, val.parquet, test.parquet
    temporal_holdout.parquet — every record on or after HOLDOUT_FROM, held back
        from all three splits. It is newer than anything the model trains on,
        so it measures whether the model holds up on future records.
    wuc_mapping.json  (label -> id, derived from train+val only)

Run on the GPU box:
    python training/prepare_data.py

Score the deployed model on the temporal holdout:
    WUC_MODEL_PATH=./wuc-model-hier python training/batch_predict.py --input data_splits/temporal_holdout.parquet --text-col text
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_app_data import find_corpus  # noqa: E402

# =============================================================================
# Config
# =============================================================================
OUT_DIR = Path("data_splits")
# Records on/after this date never enter train/val/test. 2026-04-01 is the day
# after the deployed model's training data ends, so the holdout is clean for it
# too. Move it forward only deliberately — every model compared on the holdout
# must have been trained without it.
HOLDOUT_FROM = "2026-04-01"

TEXT_FIELDS = [
    "Discrepancy",
    "Corrective Action",
    "WCE Narrative",
    "How Mal",
    "Action Taken",
]
MIN_PER_CLASS = 5
SEED = 42


# =============================================================================
# 1. Load
# =============================================================================
def main() -> None:
    corpus_path = find_corpus()
    df = pd.read_csv(corpus_path, low_memory=False)
    print(f"Corpus: {len(df):,} rows, {len(df.columns)} cols  ({corpus_path.name})")

    # =========================================================================
    # 2. Only the text fields and the label are used below. The corpus's
    #    lookup columns (SYSTEM, NOUN, ...) are derived from the WUC and would
    #    leak the label — they are never model input.
    # =========================================================================

    # =========================================================================
    # 3. LABEL HYGIENE — Corrected WUC is QC-validated ground truth
    # =========================================================================
    df["Corrected WUC"] = df["Corrected WUC"].astype(str).str.upper().str.strip()
    valid_pattern = re.compile(r"^[A-Z0-9]{3,6}$")
    mask = df["Corrected WUC"].str.match(valid_pattern, na=False)
    print(f"Dropping {(~mask).sum():,} rows with invalid Corrected WUC")
    df = df[mask].copy()

    # =========================================================================
    # 4. TEXT INPUT — combine human-written fields with [SEP]
    # =========================================================================
    def build_text(row: pd.Series) -> str:
        parts: list[str] = []
        for col in TEXT_FIELDS:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                parts.append(str(v).strip())
        return " [SEP] ".join(parts)

    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() >= 10].copy()
    print(f"After text construction: {len(df):,} rows")

    # =========================================================================
    # 4b. TEMPORAL HOLDOUT — split off BEFORE dedup and the rare-class filter,
    #     so it keeps every future record, including WUCs the model cannot
    #     emit (batch_predict reports those as unanswerable).
    # =========================================================================
    is_future = pd.to_datetime(df["Start Date"]) >= pd.Timestamp(HOLDOUT_FROM)
    holdout_df = df[is_future].copy()
    df = df[~is_future].copy()
    print(f"Temporal holdout (Start Date >= {HOLDOUT_FROM}): {len(holdout_df):,} rows "
          f"-> {len(df):,} left for train/val/test")

    # =========================================================================
    # 5. DEDUPLICATION — exact (text, label) duplicates leak between splits
    # =========================================================================
    before = len(df)
    df = df.drop_duplicates(subset=["text", "Corrected WUC"]).copy()
    print(f"After dedup: {len(df):,} rows ({before - len(df):,} removed)")

    # =========================================================================
    # 6. RARE-CLASS FILTER
    # =========================================================================
    counts = df["Corrected WUC"].value_counts()
    keep = counts[counts >= MIN_PER_CLASS].index
    df = df[df["Corrected WUC"].isin(keep)].copy()
    print(f"After rare-class filter (min {MIN_PER_CLASS}): {len(df):,} rows | "
          f"{df['Corrected WUC'].nunique():,} classes")
    print(f"Class freq -> median: {counts.median():.0f}, "
          f"mean: {counts.mean():.1f}, max: {counts.max()}")

    # =========================================================================
    # 7. 80/10/10 SPLIT — first split stratified, second random
    #    Stratifying the second split fails when a class has only 1 sample in
    #    the 20% temp set (e.g. classes with exactly 5 total examples).
    #    The temp set is already class-balanced from the first split, so a
    #    random 50/50 inside it stays approximately balanced.
    # =========================================================================
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, stratify=df["Corrected WUC"], random_state=SEED
    )
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=SEED)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Classes in train: {train_df['Corrected WUC'].nunique():,}")
    print(f"Classes in val:   {val_df['Corrected WUC'].nunique():,}")
    print(f"Classes in test:  {test_df['Corrected WUC'].nunique():,}")

    # =========================================================================
    # 8. LABEL MAP — train+val only (test stays unseen until eval)
    # =========================================================================
    labels = sorted(pd.concat([train_df, val_df])["Corrected WUC"].unique())
    wuc_to_id = {w: i for i, w in enumerate(labels)}

    # =========================================================================
    # 9. SAVE
    # =========================================================================
    OUT_DIR.mkdir(exist_ok=True)
    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUT_DIR / "val.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
    holdout_df.to_parquet(OUT_DIR / "temporal_holdout.parquet", index=False)
    with open(OUT_DIR / "wuc_mapping.json", "w") as f:
        json.dump(wuc_to_id, f, indent=2)
    print(f"Saved to {OUT_DIR}/")

    # =========================================================================
    # 10. HEALTH CHECKS — eyeball before training
    # =========================================================================
    print("\n=== Top 10 most frequent WUCs ===")
    print(df["Corrected WUC"].value_counts().head(10).to_string())
    print("\n=== Text length (chars) ===")
    print(df["text"].str.len().describe().to_string())
    print(f"\nRows with very short text (<50 chars): {(df['text'].str.len() < 50).sum()}")
    print("\n=== System-level distribution (first 2 chars of WUC) ===")
    print(df["Corrected WUC"].str[:2].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
