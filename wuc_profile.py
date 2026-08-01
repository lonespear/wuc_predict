"""Deterministic profile of a single WUC: why / when / where / lifecycle."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from base_geo import geolocate
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


def _flight_hour_buckets(
    series: pd.Series, reference: pd.Series | None = None
) -> dict[str, int]:
    """Bucket airframe flight hours into quartiles.

    `reference` supplies the distribution the quartile edges are derived from
    — pass the FULL dataset, not the WUC subset. Deriving edges from the
    subset and then binning that same subset is tautological: each bucket
    gets ~25% by construction, so every WUC looks age-neutral and genuine
    skew toward high- or low-time airframes is mathematically invisible.

    With fleet-wide edges, departure from 25% is the signal.
    """
    hrs = pd.to_numeric(series, errors="coerce").dropna()
    if hrs.empty:
        return {}

    basis = hrs if reference is None else pd.to_numeric(reference, errors="coerce").dropna()
    if basis.empty:
        basis = hrs
    q = basis.quantile([0.25, 0.5, 0.75])
    # Degenerate edges (heavy ties) would make pd.cut raise; fall back to the
    # subset's own distribution rather than dropping the section entirely.
    if len({q[0.25], q[0.5], q[0.75]}) < 3:
        q = hrs.quantile([0.25, 0.5, 0.75])
        if len({q[0.25], q[0.5], q[0.75]}) < 3:
            return {}
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


def _year_histogram(df: pd.DataFrame) -> dict[int, int]:
    """Records per calendar year.

    Mirrors _month_histogram: prefer a pre-computed YEAR column, fall back to
    deriving it from Start Date. The YEAR/MONTH columns only exist in the
    richer of the two source extracts, so the schema intersection in
    build_app_data.py drops them — without the fallback this returned {} and
    the year-over-year chart silently never rendered.
    """
    if "YEAR" in df.columns and df["YEAR"].notna().any():
        return df["YEAR"].dropna().astype(int).value_counts().sort_index().to_dict()
    if "Start Date" in df.columns:
        dates = pd.to_datetime(df["Start Date"], errors="coerce").dropna()
        if not dates.empty:
            return dates.dt.year.value_counts().sort_index().to_dict()
    return {}


def _labor_stats(subset: pd.DataFrame, full_df: pd.DataFrame) -> dict[str, Any]:
    """Maintenance burden in man-hours, and how it compares to the fleet.

    `Labor` is man-hours per maintenance action (median 2.0, IQR 1-5 across
    162,565 records, no nulls). Raw counts answer "how often"; this answers
    "how much work", which is the question a maintenance manager actually has.

    burden_index is the key number: share of fleet labor divided by share of
    fleet records. Above 1.0 means each occurrence of this WUC costs more than
    an average maintenance action — a code that is rare but expensive scores
    high, and one that is common but trivial scores low.

    NOTE: aircraft downtime is deliberately absent. Stop Date equals Start
    Date on 99.3% of records, so this dataset cannot support a down-time
    claim. See GLIDEPATH.md Phase 4 column audit.
    """
    if "Labor" not in subset.columns or "Labor" not in full_df.columns:
        return {}
    lab = pd.to_numeric(subset["Labor"], errors="coerce").dropna()
    fleet = pd.to_numeric(full_df["Labor"], errors="coerce").dropna()
    if lab.empty or fleet.empty:
        return {}

    total = float(lab.sum())
    fleet_total = float(fleet.sum())
    out: dict[str, Any] = {
        "total_man_hours": round(total, 1),
        "mean_hours_per_event": round(float(lab.mean()), 2),
        "median_hours_per_event": round(float(lab.median()), 2),
        "max_hours_single_event": round(float(lab.max()), 1),
        "fleet_mean_hours_per_event": round(float(fleet.mean()), 2),
    }
    if fleet_total > 0 and len(fleet):
        labor_share = total / fleet_total
        record_share = len(lab) / len(fleet)
        out["share_of_fleet_labor_pct"] = round(100.0 * labor_share, 3)
        out["share_of_fleet_records_pct"] = round(100.0 * record_share, 3)
        if record_share > 0:
            out["burden_index"] = round(labor_share / record_share, 2)
    return out


def _base_concentration(subset: pd.DataFrame, full_df: pd.DataFrame,
                        top_n: int = 8) -> dict[str, Any]:
    """Which bases are genuine outliers, rather than merely large.

    Raw base counts always rank the biggest bases first, which says nothing.
    This compares observed records against what the base's overall share of
    fleet maintenance would predict. index > 1.5 means the problem really is
    concentrated there; index near 1.0 means the base just does more of
    everything.
    """
    if "Base" not in subset.columns or "Base" not in full_df.columns:
        return {}
    sub = subset["Base"].astype(str).str.strip().str.title().value_counts()
    fleet = full_df["Base"].astype(str).str.strip().str.title().value_counts()
    n_sub, n_fleet = int(sub.sum()), int(fleet.sum())
    if not n_sub or not n_fleet:
        return {}

    out: dict[str, Any] = {}
    for base, count in sub.head(top_n).items():
        expected = n_sub * (int(fleet.get(base, 0)) / n_fleet)
        if expected <= 0:
            continue
        out[base] = {
            "records": int(count),
            "expected_if_typical": round(expected, 1),
            "index": round(count / expected, 2),
        }
    return out


def _trend(year_histogram: dict, date_range: tuple | None) -> dict[str, Any]:
    """Direction and size of the year-over-year trend.

    Excludes the final year when the data ends before December — the extracts
    run to 2026-03-31, so counting 2026 against full years would manufacture a
    collapse that is really just a partial year.
    """
    if not year_histogram or len(year_histogram) < 3:
        return {}
    years = sorted(int(y) for y in year_histogram)
    counts = {int(y): int(c) for y, c in year_histogram.items()}

    partial = None
    if date_range:
        try:
            last = pd.to_datetime(date_range[1])
            if last.year == years[-1] and (last.month, last.day) != (12, 31):
                partial = years[-1]
                years = years[:-1]
        except (ValueError, TypeError):
            pass
    if len(years) < 3:
        return {}

    half = len(years) // 2
    early = sum(counts[y] for y in years[:half])
    late = sum(counts[y] for y in years[-half:])
    out: dict[str, Any] = {
        "first_full_year": years[0],
        "last_full_year": years[-1],
        "earlier_period_records": early,
        "recent_period_records": late,
    }
    if early > 0:
        change = 100.0 * (late - early) / early
        out["change_pct"] = round(change, 1)
        out["direction"] = ("rising" if change > 15 else
                            "falling" if change < -15 else "steady")
    if partial:
        out["excluded_partial_year"] = partial
    return out


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
        "base_geo": [],
        "base_geo_coverage": None,
        "month_histogram": {},
        "year_histogram": {},
        "year_month_matrix": [],
        "flight_hour_buckets": {},
        "when_discovered_phase": {},
        "maint_type_phase": {},
        "cooccurring_wucs": {},
        "affected_tails": 0,
        "labor": {},
        "base_concentration": {},
        "trend": {},
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
        base_counts = subset["Base"].astype(str).str.strip().value_counts()
        profile["base_distribution"] = (
            base_counts.rename(lambda b: b.title()).head(10).to_dict()
        )
        geo: dict[str, dict[str, Any]] = {}
        mapped = 0
        for base_raw, cnt in base_counts.items():
            hit = geolocate(base_raw)
            if hit is None:
                continue
            name, lat, lon = hit
            mapped += int(cnt)
            if name in geo:
                geo[name]["count"] += int(cnt)
            else:
                geo[name] = {"base": name, "lat": lat, "lon": lon, "count": int(cnt)}
        profile["base_geo"] = sorted(geo.values(), key=lambda r: r["count"], reverse=True)
        total_base = int(base_counts.sum())
        profile["base_geo_coverage"] = (mapped, total_base)

    profile["month_histogram"] = _month_histogram(subset)

    profile["year_histogram"] = _year_histogram(subset)

    # Phase 4.1 — burden and relative measures. Counts say how often;
    # these say how much it costs, whether it is growing, and where it is
    # genuinely concentrated rather than merely where the big bases are.
    profile["labor"] = _labor_stats(subset, df)
    profile["base_concentration"] = _base_concentration(subset, df)
    profile["trend"] = _trend(profile["year_histogram"], profile["date_range"])

    # Year x month matrix for a calendar heatmap (records: {year, month, count}).
    if "Start Date" in subset.columns:
        d = pd.to_datetime(subset["Start Date"], errors="coerce").dropna()
        if not d.empty:
            mat = (
                pd.DataFrame({"year": d.dt.year, "month": d.dt.month})
                .value_counts()
                .reset_index(name="count")
                .sort_values(["year", "month"])
            )
            profile["year_month_matrix"] = mat.to_dict("records")

    if "Flight Hours" in subset.columns:
        # Quartile edges come from the whole fleet so departure from 25% means
        # something; see _flight_hour_buckets.
        profile["flight_hour_buckets"] = _flight_hour_buckets(
            subset["Flight Hours"],
            reference=df["Flight Hours"] if "Flight Hours" in df.columns else None,
        )

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
