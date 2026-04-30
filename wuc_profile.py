"""Deterministic profile of a single WUC: why / when / where / lifecycle."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from data_config import WHEN_DISCOVERED_PHASE, TYPE_MAINT_PHASE


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "was", "were", "be", "been", "has", "have", "had", "with", "that",
    "this", "it", "as", "by", "from", "/", "-", "(/)", "inop", "ops",
}


def _normalize_text(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper()
    s = s.str.replace(r"\(/\)", " ", regex=True)
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _top_phrases(series: pd.Series, n: int = 5) -> list[tuple[str, int]]:
    cleaned = _normalize_text(series)
    cleaned = cleaned[cleaned.str.len() > 3]
    return list(cleaned.value_counts().head(n).items())


def _top_keywords(series: pd.Series, n: int = 10) -> list[tuple[str, int]]:
    cleaned = _normalize_text(series)
    counter: Counter[str] = Counter()
    for text in cleaned:
        for tok in text.split():
            tl = tok.lower()
            if len(tl) < 3 or tl in STOPWORDS or tl.isdigit():
                continue
            counter[tl] += 1
    return counter.most_common(n)


def _flight_hour_buckets(series: pd.Series) -> dict[str, int]:
    hrs = pd.to_numeric(series, errors="coerce").dropna()
    if hrs.empty:
        return {}
    q = hrs.quantile([0.25, 0.5, 0.75])
    buckets = pd.cut(
        hrs,
        bins=[-1, q[0.25], q[0.5], q[0.75], float("inf")],
        labels=[
            f"Low (<{q[0.25]:.0f} hrs)",
            f"Mid-Low ({q[0.25]:.0f}-{q[0.5]:.0f})",
            f"Mid-High ({q[0.5]:.0f}-{q[0.75]:.0f})",
            f"High (>{q[0.75]:.0f} hrs)",
        ],
    )
    return buckets.value_counts().sort_index().to_dict()


def _month_histogram(df: pd.DataFrame) -> dict[str, int]:
    if "MONTH" in df.columns and df["MONTH"].notna().any():
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        counts = df["MONTH"].astype(str).str.strip().str.title().value_counts()
        return {m: int(counts.get(m, 0)) for m in order}
    if "Start Date" in df.columns:
        dates = pd.to_datetime(df["Start Date"], errors="coerce").dropna()
        return dates.dt.month_name().str[:3].value_counts().to_dict()
    return {}


def _phase_from_code(series: pd.Series, code_map: dict[str, str]) -> dict[str, int]:
    if series.empty:
        return {}
    mapped = series.astype(str).str.strip().str.upper().map(
        lambda c: code_map.get(c, f"Unknown ({c})" if c and c != "NAN" else "Unknown")
    )
    return mapped.value_counts().to_dict()


def _cooccurring_wucs(df: pd.DataFrame, full_df: pd.DataFrame, n: int = 5) -> dict[str, int]:
    if "JCN" not in df.columns or "Corrected WUC" not in full_df.columns:
        return {}
    jcns = df["JCN"].dropna().unique()
    if len(jcns) == 0:
        return {}
    same_jcn = full_df[full_df["JCN"].isin(jcns)]
    target_wucs = set(df["Corrected WUC"].astype(str).str.upper())
    others = (
        same_jcn["Corrected WUC"]
        .astype(str)
        .str.upper()
        .loc[lambda s: ~s.isin(target_wucs)]
        .value_counts()
        .head(n)
    )
    return others.to_dict()


def build_profile(
    df: pd.DataFrame,
    wuc: str,
    desc_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a structured profile dict for a single WUC."""
    wuc = wuc.strip().upper()
    col = "Corrected WUC" if "Corrected WUC" in df.columns else "WUC"
    subset = df[df[col].astype(str).str.strip().str.upper() == wuc].copy()

    profile: dict[str, Any] = {
        "wuc": wuc,
        "description": (desc_map or {}).get(wuc, "Unknown"),
        "total_records": int(len(subset)),
        "date_range": None,
        "top_discrepancy_phrases": [],
        "top_discrepancy_keywords": [],
        "top_corrective_actions": [],
        "top_corrective_keywords": [],
        "base_distribution": {},
        "month_histogram": {},
        "year_histogram": {},
        "flight_hour_buckets": {},
        "when_discovered_phase": {},
        "maint_type_phase": {},
        "cooccurring_wucs": {},
        "affected_tails": 0,
    }

    if subset.empty:
        return profile

    if "Start Date" in subset.columns:
        dates = pd.to_datetime(subset["Start Date"], errors="coerce").dropna()
        if not dates.empty:
            profile["date_range"] = (
                dates.min().strftime("%Y-%m-%d"),
                dates.max().strftime("%Y-%m-%d"),
            )

    if "Discrepancy" in subset.columns:
        profile["top_discrepancy_phrases"] = _top_phrases(subset["Discrepancy"], 5)
        profile["top_discrepancy_keywords"] = _top_keywords(subset["Discrepancy"], 10)

    if "Corrective Action" in subset.columns:
        profile["top_corrective_actions"] = _top_phrases(subset["Corrective Action"], 5)
        profile["top_corrective_keywords"] = _top_keywords(subset["Corrective Action"], 10)

    if "Base" in subset.columns:
        profile["base_distribution"] = (
            subset["Base"].astype(str).str.strip().str.title().value_counts().head(10).to_dict()
        )

    profile["month_histogram"] = _month_histogram(subset)

    if "YEAR" in subset.columns:
        profile["year_histogram"] = (
            subset["YEAR"].dropna().astype(int).value_counts().sort_index().to_dict()
        )

    if "Flight Hours" in subset.columns:
        profile["flight_hour_buckets"] = _flight_hour_buckets(subset["Flight Hours"])

    if "When Discovered Code" in subset.columns:
        profile["when_discovered_phase"] = _phase_from_code(
            subset["When Discovered Code"], WHEN_DISCOVERED_PHASE
        )

    if "Type Maint Code" in subset.columns:
        profile["maint_type_phase"] = _phase_from_code(
            subset["Type Maint Code"], TYPE_MAINT_PHASE
        )

    profile["cooccurring_wucs"] = _cooccurring_wucs(subset, df, 5)

    if "Tail Number" in subset.columns:
        profile["affected_tails"] = int(subset["Tail Number"].nunique())

    return profile
