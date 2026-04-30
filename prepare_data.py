"""Merge two raw KC-135 maintenance extracts into clean train/val/test splits
for WUC prediction.

Inputs (configurable below):
    PATH_A — richer 30-col enriched extract (e.g. FinalData.csv)
    PATH_B — leaner 21-col raw extract

Outputs (in ./data_splits/):
    train.parquet, val.parquet, test.parquet
    wuc_mapping.json  (label -> id, derived from train+val only)

Run on the GPU box:
    python prepare_data.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# Config
# =============================================================================
PATH_A = "FinalData.csv"
PATH_B = "new_data.csv"            # rename to your second file
OUT_DIR = Path("data_splits")

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
    df_a = pd.read_csv(PATH_A, low_memory=False)
    df_b = pd.read_csv(PATH_B, low_memory=False)
    print(f"A: {len(df_a):,} rows, {len(df_a.columns)} cols")
    print(f"B: {len(df_b):,} rows, {len(df_b.columns)} cols")

    # =========================================================================
    # 2. Reduce A to B's schema (drop derived/enriched columns)
    #    Those extras (YEAR, MONTH, SYSTEM, ...) are derivable from base fields
    #    and including them risks leaking the label (SYSTEM is from WUC).
    # =========================================================================
    common = [c for c in df_b.columns if c in df_a.columns]
    df = pd.concat([df_a[common], df_b[common]], ignore_index=True)
    print(f"Common cols: {len(common)} | Combined: {len(df):,} rows")
    print(f"Dropped from A: {sorted(set(df_a.columns) - set(common))}")

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
    print(f"Class freq → median: {counts.median():.0f}, "
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
