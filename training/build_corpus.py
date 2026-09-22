"""Build the single deduplicated KC-135 maintenance corpus from every raw extract.

This is the one file everyone should work from. It stacks all the raw extracts,
removes duplicates, and resolves the case where the same job was re-scrubbed
between extracts.

Raw extracts, named data_<delivery order>_<coverage>. The single master copy
lives in OneDrive "Documents/kc135/data/" (see README.md there for the full
provenance). Copy that folder into --raw-dir; all of it is CUI and gitignored.

    data_1_2019-01_2024-10.csv     127,505 rows  31 cols  received by 2025-01
    data_2_2019-01_2026-03.csv     132,962 rows  21 cols  received 2026-04-24
    data_3_2025-02_2026-07.xlsx     25,027 rows  21 cols  received 2026-09-21
    reference/wuc_codes_{after,before}_scrub.csv  WUC tables from delivery 2

    Never build from a CSV that has been opened and re-saved in Excel. A copy of
    data_1 treated that way (deleted 2026-09-22) had codes turned into numbers
    or dates in 1,281 rows (14E00 -> 1.40E+01, 24DEC -> 24-Dec).

Dedup rules
  1. Exact duplicates across the 21 shared columns are dropped.
  2. A job is (JCN, WCE ID, Tail Number). When the same job appears in more than
     one extract with different values, the later extract is a re-scrub of the
     same record (corrected WUCs, fixed typos, updated flight hours), so the
     NEWEST extract wins. Losing versions go to combined_*_superseded.csv.
  3. Repeated keys *within* one extract are kept; the extract reports them as
     separate rows.

Rule 2 matters for training. data_1 + data_2 without it hold ~6,000 jobs twice,
~1,700 of them with two different Corrected WUC labels — so the same job can sit
in train and test at once, under different labels.

data_1's 10 lookup columns (SYSTEM, NOUN, ...) are re-derived for every row from
the winning codes and checked against data_1's own values. They are derived from
the label — never feed them to the classifier as input.

Usage:
    python training/build_corpus.py                      # reads data/, writes data/
    python training/build_corpus.py \
        --raw-dir ".../OneDrive - West Point/Documents/kc135/data" \
        --out-dir ".../OneDrive - West Point/Documents/kc135/data/combined"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_app_data import _parse_dates  # noqa: E402  (same date rules as the app)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Oldest -> newest. On conflicting versions of one job, the later source wins.
SOURCES = [
    ("data_1", "data_1_2019-01_2024-10.csv", None),
    ("data_2", "data_2_2019-01_2026-03.csv", None),
    ("data_3", "data_3_2025-02_2026-07.xlsx", "DATA"),
]
# WUC tables from delivery 2, exported from its "AFTER SCURB" / "BEFORE SCRUB"
# tabs. After-scrub takes precedence.
REF_FILES = ["reference/wuc_codes_after_scrub.csv", "reference/wuc_codes_before_scrub.csv"]

KEY = ["JCN", "WCE ID", "Tail Number"]
INT_COLS = ["How Mal", "WCE ID", "Units Produced", "How Mal Class"]
CODE_COLS = ["WUC", "Corrected WUC", "Action Taken", "Corrected Action Taken",
             "When Discovered Code", "Type Maint Code"]
BY_CWUC = ["CORRECTED INTEGRITY PROGRAM", "CORRECTED NOUN",
           "SUBSYSTEM", "SPECIFIC SYSTEM"]
BY_WUC = ["NOUN", "INTEGRITY PROGRAM"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name: str, path: Path, sheet: str | None) -> pd.DataFrame:
    df = (pd.read_csv(path, low_memory=False) if sheet is None
          else pd.read_excel(path, sheet_name=sheet))
    for c in ["Start Date", "Stop Date"]:
        df[c] = _parse_dates(df[c]).dt.strftime("%Y-%m-%d")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Codes are uppercase by convention; a handful arrive lowercase (13fc0).
    for c in CODE_COLS:
        low = df[c].notna() & (df[c] != df[c].str.upper())
        if low.any():
            print(f"    uppercased {int(low.sum())} {c} value(s)")
        df[c] = df[c].str.upper()
    df["source"] = name
    print(f"  {name:8s} {len(df):>8,} rows  {df.shape[1] - 1} cols  "
          f"{df['Start Date'].min()} -> {df['Start Date'].max()}")
    return df


def build_lookups(d1: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WUC -> descriptor tables: data_1 first, delivery-2 WUC tables as fallback."""
    by_c = d1.groupby("Corrected WUC")[BY_CWUC].first()
    by_w = d1.groupby("WUC")[BY_WUC].first()
    ref_paths = [raw_dir / f for f in REF_FILES]
    if not all(p.exists() for p in ref_paths):
        print("  NOTE: reference/ WUC tables not found; codes unseen in data_1 get no noun")
        return by_c, by_w
    ref = pd.concat([pd.read_csv(p, dtype={"WUC": str}) for p in ref_paths])
    ref["WUC"] = ref["WUC"].astype(str).str.strip().str.upper()
    ref = ref.drop_duplicates("WUC").set_index("WUC")
    fb_c = pd.DataFrame({"CORRECTED INTEGRITY PROGRAM": ref["INTEGRITY PROGRAM"],
                         "CORRECTED NOUN": ref["NOUN"]})
    fb_w = pd.DataFrame({"NOUN": ref["NOUN"],
                         "INTEGRITY PROGRAM": ref["INTEGRITY PROGRAM"]})
    return by_c.combine_first(fb_c), by_w.combine_first(fb_w)


def derive(df: pd.DataFrame, by_c: pd.DataFrame, by_w: pd.DataFrame) -> pd.DataFrame:
    cw = df["Corrected WUC"]
    out = by_c.reindex(cw).reset_index(drop=True)
    # The hierarchy is positional in the code; SYSTEM is always the 2-char
    # prefix (100% on data_1). Fill SUBSYSTEM/SPECIFIC for codes data_1 never saw.
    out["SYSTEM"] = cw.str[:2].values
    for col, n in [("SUBSYSTEM", 3), ("SPECIFIC SYSTEM", 4)]:
        out[col] = out[col].astype("string").fillna(cw.str[:n].reset_index(drop=True))
    out[BY_WUC] = by_w.reindex(df["WUC"]).reset_index(drop=True)
    out["WUC CORRECTIONS"] = (df["WUC"] == cw).map({True: "GOOD", False: "CORRECTED"}).values
    sd = pd.to_datetime(df["Start Date"])
    out["YEAR"] = sd.dt.year.astype("Int64").values
    out["MONTH"] = sd.dt.strftime("%b").values
    return out


def _key_hash(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.util.hash_pandas_object(df[KEY].astype(str), index=False).values,
                     index=df.index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data",
                    help="folder holding the raw extracts (default: data/)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data",
                    help="where to write combined_*.csv (default: data/, gitignored)")
    args = ap.parse_args()

    paths = {n: args.raw_dir / f for n, f, _ in SOURCES}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("ERROR: missing raw extract(s):\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    print(f"Loading from {args.raw_dir}:")
    frames = [load(n, paths[n], s) for n, _, s in SOURCES]
    d1 = frames[0]
    common = [c for c in frames[1].columns if c != "source"]
    extras = [c for c in d1.columns if c not in common and c != "source"]
    rank = {n: i for i, (n, _, _) in enumerate(SOURCES)}

    stacked = pd.concat([f[common + ["source"]] for f in frames], ignore_index=True)
    stacked["_rank"] = stacked["source"].map(rank)
    n_raw = len(stacked)

    # Rule 1 — exact dedup, remembering every extract a row appeared in.
    stacked["_h"] = pd.util.hash_pandas_object(stacked[common].astype(str), index=False).values
    seen_in = stacked.groupby("_h")["source"].agg(lambda s: "+".join(dict.fromkeys(s)))
    stacked = stacked.sort_values("_rank", ascending=False).drop_duplicates("_h").copy()
    stacked["sources"] = stacked["_h"].map(seen_in)
    n_exact = len(stacked)

    # Rule 2 — cross-extract re-scrubs of the same job: newest wins.
    stacked["_k"] = _key_hash(stacked)
    multi = stacked[stacked.groupby("_k")["source"].transform("nunique") > 1]
    losers = multi[multi["_rank"] < multi.groupby("_k")["_rank"].transform("max")]
    corpus = stacked.drop(losers.index)
    assert corpus.groupby("_k")["source"].nunique().max() == 1  # rule 3 holds

    by_c, by_w = build_lookups(d1, args.raw_dir)
    chk = derive(d1, by_c, by_w)
    print("\nDerived lookup columns vs data_1's originals:")
    for c in extras:
        a = float((chk[c].astype(str).values == d1[c].astype(str).values).mean())
        # SUBSYSTEM / SPECIFIC SYSTEM miss by 8 rows: data_1 derived them from
        # lowercase codes (13fc0 -> 13f) before the uppercase fix above.
        print(f"  {'OK  ' if a > 0.9999 else 'WARN'} {c:28s} {a:.4%}")

    corpus = corpus.sort_values(["Start Date", "Tail Number", "JCN", "WCE ID"]).reset_index(drop=True)
    corpus = pd.concat([corpus[common], derive(corpus, by_c, by_w)[extras],
                        corpus[["source", "sources"]]], axis=1)

    winner = corpus.assign(_k=_key_hash(corpus)).groupby("_k")["source"].first()
    losers = (losers.assign(superseded_by=losers["_k"].map(winner))
              [common + ["source", "sources", "superseded_by"]].sort_values(KEY))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = corpus["Start Date"].min()[:7], corpus["Start Date"].max()[:7]
    out = args.out_dir / f"combined_{lo}_{hi}.csv"
    out_sup = args.out_dir / f"combined_{lo}_{hi}_superseded.csv"
    corpus.to_csv(out, index=False)
    losers.to_csv(out_sup, index=False)

    # Counts and hashes only — safe to paste into a report or commit message,
    # and lets two people confirm they built from the same inputs.
    manifest = {
        "rows": len(corpus), "cols": corpus.shape[1],
        "unique_jobs": int(corpus[KEY].drop_duplicates().shape[0]),
        "start_date_min": corpus["Start Date"].min(),
        "start_date_max": corpus["Start Date"].max(),
        "rows_stacked": n_raw, "rows_after_exact_dedup": n_exact,
        "rows_superseded": len(losers),
        "winning_source": corpus["source"].value_counts().to_dict(),
        "inputs": {p.name: _sha256(p) for p in paths.values()},
        "corpus_sha256": _sha256(out),
    }
    (args.out_dir / f"combined_{lo}_{hi}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nRows stacked from all extracts : {n_raw:>8,}")
    print(f"After exact dedup              : {n_exact:>8,}  (-{n_raw - n_exact:,})")
    print(f"After newest-wins on re-scrubs : {len(corpus):>8,}  (-{len(losers):,} -> {out_sup.name})")
    print(f"\nWrote {out}: {corpus.shape[0]:,} x {corpus.shape[1]}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"      {corpus['Start Date'].min()} -> {corpus['Start Date'].max()}")
    print(f"      corpus sha256 {manifest['corpus_sha256'][:16]}...  (full hash in the _manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
