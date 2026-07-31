"""Build the app's operational dataset from the raw extracts.

Why this exists
---------------
`main_app.py` gets its dataframe from `data_config.resolve_data_path()`, which
resolved to `FinalData.csv` at repo root — a **stale artifact** that was an
*input* to an earlier version of `prepare_data.py` (see commit `370fd3a`, which
repointed PATH_A at `data/data1.csv`). It carries 20 columns and is missing
`Base`, `Flight Hours`, `JCN`, `When Discovered Code` and `Type Maint Code`.

`wuc_profile.py` guards each of those with `if "X" in df.columns`, so they were
silently omitted rather than raising — leaving six sections of the WUC profile
permanently empty: base_distribution, base_geo, flight_hour_buckets,
cooccurring_wucs, when_discovered_phase, maint_type_phase. That is also why the
Tab 3 map never drew bubbles (they key off `Base`) and why the sectioned
analyst prompt kept reporting "insufficient data".

The real extracts in `data/` have every one of those columns. This script
merges them into `app_data.csv`, which `data_config.py` now prefers over
`FinalData.csv`.

Training is unaffected — `prepare_data.py` still writes `data_splits/` for that.

Rollback: delete `app_data.csv`; `resolve_data_path()` falls back to
`FinalData.csv` and you are exactly where you started.

Usage:
    python training/build_app_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_A = REPO_ROOT / "data" / "data1.csv"
PATH_B = REPO_ROOT / "data" / "data2.csv"
OUT_PATH = REPO_ROOT / "app_data.csv"

# Columns the WUC profile and Tab 2 reach for. Reported explicitly after the
# merge so a silently-missing column can never go unnoticed again.
PROFILE_COLUMNS = [
    "Corrected WUC",
    "Discrepancy",
    "Corrective Action",
    "Tail Number",
    "Start Date",
    "Base",
    "Flight Hours",
    "JCN",
    "When Discovered Code",
    "Type Maint Code",
]

# sum_utils.py:270-271 prefers these, falling back to the raw columns with `or`.
# Regenerating them restores phrase grouping in Tab 2.
NORMALIZE_MAP = {
    "Discrepancy": "discrepancy_normalized",
    "Corrective Action": "corrective_action_normalized",
}

DATE_COLUMNS = ["Start Date", "Stop Date"]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse a date column that the raw extracts store as a NUMBER.

    `FinalData.csv` shipped pre-formatted dates. The raw extracts do not, and
    `pd.to_datetime` on an integer column interprets it as *nanoseconds since
    epoch* — so every value lands within a fraction of a second of 1970-01-01,
    all in January. That silently empties every date filter in Tab 2
    (sum_utils.py:180 coerces, so unparseable becomes NaT and every
    `NaT >= start_date` is False) and flattens Tab 3's date_range,
    month_histogram, year_histogram and calendar heatmap.

    Detects the encoding rather than assuming it, so this keeps working if a
    future extract arrives in a different format.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()

    if len(valid):
        # YYYYMMDD integers, e.g. 20240115
        if valid.between(19000101, 21001231).mean() > 0.9:
            return pd.to_datetime(
                numeric.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
            )
        # Excel serial days since the 1900 epoch (offset by its leap-year bug)
        if valid.between(1, 80000).mean() > 0.9:
            return pd.to_datetime(
                numeric, unit="D", origin="1899-12-30", errors="coerce"
            )

    return pd.to_datetime(series, errors="coerce")


def _normalize(series: pd.Series) -> pd.Series:
    """Collapse trivial write-up variants so top-N phrase counts group properly.

    Source text is 100% uppercase (audited 2026-07-31, n=132,962), so this
    only has to fix whitespace and trailing punctuation.
    """
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.rstrip(". ")
    )


def main() -> int:
    for path in (PATH_A, PATH_B):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    df_a = pd.read_csv(PATH_A, low_memory=False)
    df_b = pd.read_csv(PATH_B, low_memory=False)
    print(f"A: {len(df_a):,} rows, {len(df_a.columns)} cols  ({PATH_A.name})")
    print(f"B: {len(df_b):,} rows, {len(df_b.columns)} cols  ({PATH_B.name})")

    # Same schema rule as prepare_data.py: reduce to the columns both share.
    # data2 is a strict subset of data1, so this keeps all 21 of data2's
    # columns and drops only data1's 10 derived extras (SYSTEM, NOUN, YEAR...).
    common = [c for c in df_b.columns if c in df_a.columns]
    dropped = sorted(set(df_a.columns) - set(common))
    print(f"Common schema: {len(common)} cols")
    print(f"Dropped from A: {dropped}")

    merged = pd.concat([df_a[common], df_b[common]], ignore_index=True)
    before = len(merged)

    # Row-level exact dedup only. Deliberately NOT the (text, label) dedup
    # prepare_data.py uses — Tab 2 and Tab 3 count maintenance records, and
    # two distinct jobs can legitimately share identical write-up text.
    merged = merged.drop_duplicates().reset_index(drop=True)
    print(f"Merged: {before:,} rows -> {len(merged):,} after exact dedup "
          f"({before - len(merged):,} removed)")

    for src, dest in NORMALIZE_MAP.items():
        if src in merged.columns:
            merged[dest] = _normalize(merged[src])
            print(f"Generated {dest}")

    print("\nDates (written as ISO YYYY-MM-DD):")
    date_ok = True
    for col in DATE_COLUMNS:
        if col not in merged.columns:
            continue
        raw_dtype = merged[col].dtype
        parsed = _parse_dates(merged[col])
        pct = 100.0 * parsed.notna().mean() if len(parsed) else 0.0
        merged[col] = parsed.dt.strftime("%Y-%m-%d")
        lo = parsed.min()
        hi = parsed.max()
        span = f"{lo:%Y-%m-%d} -> {hi:%Y-%m-%d}" if parsed.notna().any() else "none parsed"
        flag = "OK     " if pct > 95 else "WARN   "
        print(f"  {flag} {col:<12} from {str(raw_dtype):<9} {pct:5.1f}% parsed   {span}")
        if pct <= 95:
            date_ok = False
        # A column that parses fine but lands in 1970 is the integer-as-
        # nanoseconds failure; catch it here rather than in the UI.
        if parsed.notna().any() and hi.year < 1980:
            print(f"  ERROR  {col} parsed entirely before 1980 — epoch misread",
                  file=sys.stderr)
            date_ok = False

    print("\nProfile-critical columns:")
    missing = []
    for col in PROFILE_COLUMNS:
        if col in merged.columns:
            non_null = int(merged[col].notna().sum())
            pct = 100.0 * non_null / len(merged) if len(merged) else 0.0
            print(f"  OK      {col:<22} {non_null:>9,} non-null ({pct:.1f}%)")
        else:
            missing.append(col)
            print(f"  MISSING {col}")

    merged.to_csv(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nWrote {OUT_PATH.name}: {len(merged):,} rows, "
          f"{len(merged.columns)} cols, {size_mb:.1f} MB")

    if missing:
        print(f"\nWARNING: {len(missing)} profile column(s) still missing: "
              f"{missing}", file=sys.stderr)
        return 1
    if not date_ok:
        print("\nWARNING: date parsing did not look clean — Tab 2 filters and "
              "Tab 3 histograms depend on it. Check the samples above.",
              file=sys.stderr)
        return 1

    print("\nRestart Streamlit to pick it up. Tab 3 should now populate "
          "base_distribution, flight_hour_buckets, cooccurring_wucs, "
          "when_discovered_phase and maint_type_phase — and the map should "
          "draw bubbles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
