import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import calendar
import streamlit as st

# Calendar-quarter style season windows (month numbers)
_SEASONS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "fall": (9, 11),
    "autumn": (9, 11),
    "winter": (12, 2),  # spans year boundary — handled specially
}

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def _month_bounds(year, month):
    """First and last calendar day of a given year/month as ISO strings."""
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def parse_user_query(query):
    """Best-effort NL → filter dict. Recognizes tail numbers, WUC codes, and a
    range of date expressions. Anything it cannot parse is simply omitted — the
    caller is expected to surface the interpreted filters back to the user.
    """
    filters = {}
    original = query.strip()
    q = original.lower()

    # --- Tail Number (has a dash, so it never collides with a WUC token) ---
    tail_match = re.search(r"\b\d{2}-\d{4}\b", q)
    if tail_match:
        filters["tail_number"] = tail_match.group(0)

    # --- WUC code -------------------------------------------------------
    # Explicit "wuc XXXXX" form first.
    wuc_match = re.search(r"wuc[:#\s]*([0-9a-z]{4,6})", q)
    if wuc_match:
        filters["wuc"] = wuc_match.group(1).upper()
    else:
        # Bare 5-char alphanumeric token that looks like a WUC: must contain at
        # least one digit (rules out English words) and not be a 4-digit year.
        for tok in re.findall(r"\b[0-9a-z]{5}\b", q):
            if any(c.isdigit() for c in tok) and not re.fullmatch(r"20\d{2}\w?", tok):
                # avoid grabbing the tail-number fragments
                if not (tail_match and tok in tail_match.group(0)):
                    filters["wuc"] = tok.upper()
                    break

    # --- Explicit "from <Mon> <Year> to <Mon> <Year>" ------------------
    rng = re.search(
        r"from\s+([a-z]+)\.?\s+(20\d{2})\s+to\s+([a-z]+)\.?\s+(20\d{2})", q
    )
    if rng:
        sm, sy, em, ey = rng.groups()
        if sm in _MONTHS and em in _MONTHS:
            filters["start_date"] = _month_bounds(int(sy), _MONTHS[sm])[0]
            filters["end_date"] = _month_bounds(int(ey), _MONTHS[em])[1]
            return filters

    # --- "from <Year> to <Year>" ---------------------------------------
    yr_rng = re.search(r"from\s*(20\d{2})\s*to\s*(20\d{2})", q)
    if yr_rng:
        sy, ey = yr_rng.groups()
        filters["start_date"] = f"{sy}-01-01"
        filters["end_date"] = f"{ey}-12-31"
        return filters

    # --- "between <Year> and <Year>" -----------------------------------
    btw = re.search(r"between\s*(20\d{2})\s*and\s*(20\d{2})", q)
    if btw:
        sy, ey = sorted(btw.groups())
        filters["start_date"] = f"{sy}-01-01"
        filters["end_date"] = f"{ey}-12-31"
        return filters

    # --- season + year, optionally "since"/"after" ---------------------
    season_m = re.search(
        r"(since|after|from|in|during)?\s*(spring|summer|fall|autumn|winter)\s+(of\s+)?(20\d{2})",
        q,
    )
    if season_m:
        lead, season, _, yr = season_m.groups()
        yr = int(yr)
        s_mon, e_mon = _SEASONS[season]
        if season == "winter":
            start = f"{yr}-12-01"
            end = _month_bounds(yr + 1, 2)[1]
        else:
            start = _month_bounds(yr, s_mon)[0]
            end = _month_bounds(yr, e_mon)[1]
        filters["start_date"] = start
        if lead in ("since", "after", "from"):
            pass  # open-ended on the right
        else:
            filters["end_date"] = end
        return filters

    # --- "since/after <Mon> <Year>" or "since/after <Year>" ------------
    since_m = re.search(r"(since|after|from)\s+(?:([a-z]+)\.?\s+)?(20\d{2})", q)
    if since_m:
        _, mon, yr = since_m.groups()
        yr = int(yr)
        if mon and mon in _MONTHS:
            filters["start_date"] = _month_bounds(yr, _MONTHS[mon])[0]
        else:
            filters["start_date"] = f"{yr}-01-01"
        # no end_date — open-ended
        return filters

    # --- "<Mon> <Year>" (single month) ---------------------------------
    single_mon = re.search(r"\b([a-z]+)\.?\s+(20\d{2})\b", q)
    if single_mon and single_mon.group(1) in _MONTHS:
        yr = int(single_mon.group(2))
        s, e = _month_bounds(yr, _MONTHS[single_mon.group(1)])
        filters["start_date"], filters["end_date"] = s, e
        return filters

    # --- "in <Year>" / bare "<Year>" -----------------------------------
    year_m = re.search(r"\b(?:in\s+)?(20\d{2})\b", q)
    if year_m:
        yr = int(year_m.group(1))
        filters["start_date"] = f"{yr}-01-01"
        filters["end_date"] = f"{yr}-12-31"

    # --- "this year" / "last year" -------------------------------------
    now = datetime.today()
    if re.search(r"\bthis year\b", q):
        filters["start_date"] = f"{now.year}-01-01"
        filters["end_date"] = f"{now.year}-12-31"
    elif re.search(r"\blast year\b", q):
        filters["start_date"] = f"{now.year - 1}-01-01"
        filters["end_date"] = f"{now.year - 1}-12-31"

    # --- "last X months / years" ---------------------------------------
    lastm = re.search(r"last\s*(\d+)\s*month", q)
    if lastm:
        n = int(lastm.group(1))
        filters["start_date"] = (now - timedelta(days=30 * n)).strftime("%Y-%m-%d")
        filters["end_date"] = now.strftime("%Y-%m-%d")
    lasty = re.search(r"last\s*(\d+)\s*year", q)
    if lasty:
        n = int(lasty.group(1))
        filters["start_date"] = (now - timedelta(days=365 * n)).strftime("%Y-%m-%d")
        filters["end_date"] = now.strftime("%Y-%m-%d")

    return filters


def describe_filters(filters):
    """Human-readable list of the filters that were actually applied."""
    parts = []
    if filters.get("tail_number"):
        parts.append(f"Tail = **{filters['tail_number']}**")
    if filters.get("wuc"):
        parts.append(f"WUC = **{filters['wuc']}**")
    s, e = filters.get("start_date"), filters.get("end_date")
    if s or e:
        lo = pd.to_datetime(s).strftime("%b %Y") if s else "earliest"
        hi = pd.to_datetime(e).strftime("%b %Y") if e else "latest"
        parts.append(f"dates **{lo} → {hi}**")
    return parts


# ----------------------------------------------------------------------
# Filtering + aggregation
# ----------------------------------------------------------------------

def query_records(df, tail_number=None, wuc=None, start_date=None, end_date=None):
    """Filter the dataframe based on provided conditions."""
    results = df.copy()

    if "Start Date" in results.columns:
        results["Start Date"] = pd.to_datetime(results["Start Date"], errors="coerce")

    if tail_number:
        results = results[
            results["Tail Number"].astype(str).str.strip() == str(tail_number).strip()
        ]
    if wuc:
        results = results[
            results["Corrected WUC"].astype(str).str.strip().str.upper()
            == str(wuc).strip().upper()
        ]
    if start_date is not None and "Start Date" in results.columns:
        results = results[results["Start Date"] >= pd.to_datetime(start_date)]
    if end_date is not None and "Start Date" in results.columns:
        results = results[results["Start Date"] <= pd.to_datetime(end_date)]

    return results


def issues_by_month(results):
    """Count filtered issues by calendar month (YYYY-MM)."""
    if results.empty or "Start Date" not in results.columns:
        return {}
    monthly = results.dropna(subset=["Start Date"]).copy()
    monthly["Month"] = monthly["Start Date"].dt.to_period("M").astype(str)
    return monthly.groupby("Month").size().sort_index().to_dict()


def seasonality(results):
    """Count issues by month-of-year (Jan..Dec) summed across all years."""
    if results.empty or "Start Date" not in results.columns:
        return {}
    s = results.dropna(subset=["Start Date"]).copy()
    by = s["Start Date"].dt.month.value_counts()
    return {calendar.month_abbr[m]: int(by.get(m, 0)) for m in range(1, 13)}


def _topn_counts(results, col, n=10):
    if results.empty or col not in results.columns:
        return {}
    vc = results[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).dropna()
    return vc.value_counts().head(n).to_dict()


def wuc_breakdown(results, desc_map=None, n=10):
    if results.empty or "Corrected WUC" not in results.columns:
        return {}
    counts = (
        results["Corrected WUC"].astype(str).str.strip().str.upper().value_counts().head(n)
    )
    out = {}
    for wuc, count in counts.items():
        label = f"{wuc} ({desc_map[wuc]})" if (desc_map and wuc in desc_map) else wuc
        out[label] = int(count)
    return out


def _prior_period_count(df, start_date, end_date, **other):
    """Count records in the equal-length window immediately before [start, end]."""
    if not start_date or not end_date:
        return None
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    span = e - s
    return len(query_records(df, start_date=s - span - pd.Timedelta(days=1), end_date=s - pd.Timedelta(days=1), **other))


def analyze_results(df, desc_map=None, **filters):
    """Run the full analysis on filtered records."""
    results = query_records(df, **filters)

    out = {
        "total_issues": len(results),
        "top_discrepancies": {},
        "top_fixes": {},
        "issues_by_month": {},
        "seasonality": {},
        "wuc_breakdown": {},
        "base_breakdown": {},
        "tail_leaderboard": {},
        "distinct_tails": 0,
        "distinct_wucs": 0,
        "date_span": None,
        "prior_period_count": None,
        "results": results,
        "filters": filters,
    }

    if len(results) == 0:
        return out

    out["top_discrepancies"] = _topn_counts(results, "discrepancy_normalized", 10) or _topn_counts(results, "Discrepancy", 10)
    out["top_fixes"] = _topn_counts(results, "corrective_action_normalized", 10) or _topn_counts(results, "Corrective Action", 10)
    out["issues_by_month"] = issues_by_month(results)
    out["seasonality"] = seasonality(results)
    out["base_breakdown"] = _topn_counts(results, "Base", 10)

    if "Tail Number" in results.columns:
        out["distinct_tails"] = results["Tail Number"].astype(str).str.strip().nunique()
        if "tail_number" not in filters:
            out["tail_leaderboard"] = _topn_counts(results, "Tail Number", 10)
    if "Corrected WUC" in results.columns:
        out["distinct_wucs"] = results["Corrected WUC"].astype(str).str.strip().str.upper().nunique()
        if "wuc" not in filters:
            out["wuc_breakdown"] = wuc_breakdown(results, desc_map, 10)

    if "Start Date" in results.columns:
        sd = results["Start Date"].dropna()
        if not sd.empty:
            out["date_span"] = (sd.min().strftime("%b %Y"), sd.max().strftime("%b %Y"))

    other = {k: v for k, v in filters.items() if k in ("tail_number", "wuc")}
    out["prior_period_count"] = _prior_period_count(
        df, filters.get("start_date"), filters.get("end_date"), **other
    )

    return out


# ----------------------------------------------------------------------
# Legacy plaintext formatter (kept for sum_app.py; no longer dumps the month list)
# ----------------------------------------------------------------------

def format_wuc_section(wuc_dict):
    if not wuc_dict:
        return []
    lines = ["Top Problem Areas (WUC):"]
    for i, (label, count) in enumerate(wuc_dict.items(), start=1):
        lines.append(f"{i}. {label:<60} {count}")
    lines.append("")
    return lines


def format_answer(analysis):
    if analysis["total_issues"] == 0:
        return "No matching records found."
    lines = [f"Total issues: {analysis['total_issues']:,}", ""]
    if analysis["top_discrepancies"]:
        lines.append("Top discrepancies:")
        for issue, count in list(analysis["top_discrepancies"].items())[:5]:
            lines.append(f"- {issue} ({count})")
        lines.append("")
    if analysis["top_fixes"]:
        lines.append("Top fixes:")
        for fix, count in list(analysis["top_fixes"].items())[:5]:
            lines.append(f"- {fix} ({count})")
        lines.append("")
    return "\n".join(lines)
