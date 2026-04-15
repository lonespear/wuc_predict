import pandas as pd
import numpy as np
import matplotlib as plt 
import re 
from datetime import datetime, timedelta
import calendar
import streamlit as st

def parse_user_query(query):
    filters = {}

    query = query.lower()

    # --- Tail Number ---
    tail_match = re.search(r"\b\d{2}-\d{4}\b", query)
    if tail_match:
        filters["tail_number"] = tail_match.group(0)

    # --- WUC ---
    wuc_match = re.search(r"wuc\s*([0-9a-zA-Z]+)", query)
    if wuc_match:
        filters["wuc"] = wuc_match.group(1).upper()

    # --- From Month Year to Month Year ---
    range_match = re.search(
        r"from\s+([a-zA-Z]+)\s+(20\d{2})\s+to\s+([a-zA-Z]+)\s+(20\d{2})",
        query
    )

    if range_match:
        start_month_str, start_year, end_month_str, end_year = range_match.groups()

        # Convert month names to numbers
        start_month = list(calendar.month_name).index(start_month_str.capitalize())
        end_month = list(calendar.month_name).index(end_month_str.capitalize())

        # Start date = first day of start month
        start_date = f"{start_year}-{start_month:02d}-01"

        # End date = last day of end month
        last_day = calendar.monthrange(int(end_year), end_month)[1]
        end_date = f"{end_year}-{end_month:02d}-{last_day}"

        filters["start_date"] = start_date
        filters["end_date"] = end_date

        # --- From YEAR to YEAR ---
    year_range_match = re.search(
        r"from\s*(20\d{2})\s*to\s*(20\d{2})",
        query
    )

    if year_range_match:
        start_year, end_year = year_range_match.groups()

        filters["start_date"] = f"{start_year}-01-01"
        filters["end_date"] = f"{end_year}-12-31"

        return filters  # highest priority

    # --- Year (e.g., "in 2025") ---
    year_match = re.search(r"in\s*(20\d{2})", query)
    if year_match:
        year = int(year_match.group(1))
        filters["start_date"] = f"{year}-01-01"
        filters["end_date"] = f"{year}-12-31"

    # --- Last X months ---
    months_match = re.search(r"last\s*(\d+)\s*month", query)
    if months_match:
        months = int(months_match.group(1))
        end_date = datetime.today()
        start_date = end_date - timedelta(days=30 * months)

        filters["start_date"] = start_date.strftime("%Y-%m-%d")
        filters["end_date"] = end_date.strftime("%Y-%m-%d")


    return filters

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
    """Count filtered issues by month."""
    if results.empty or "Start Date" not in results.columns:
        return {}

    monthly = results.copy()
    monthly = monthly.dropna(subset=["Start Date"])
    monthly["Month"] = monthly["Start Date"].dt.to_period("M").astype(str)

    return monthly.groupby("Month").size().sort_index().to_dict()

def wuc_breakdown(results, desc_map=None):
    if results.empty or "Corrected WUC" not in results.columns:
        return {}

    counts = (
        results["Corrected WUC"]
        .astype(str)
        .str.strip()
        .str.upper()
        .value_counts()
        .head(10)
    )

    output = {}

    for wuc, count in counts.items():
        if desc_map and wuc in desc_map:
            label = f"{wuc} ({desc_map[wuc]})"
            #clean_desc = clean_wuc_desc(desc_map[wuc])
            #label = f"{wuc} ({clean_desc})"
        else:
            label = wuc

        output[label] = count

    return output

def format_wuc_section(wuc_dict):
    """Format WUC breakdown in a cleaner ranked list."""
    if not wuc_dict:
        return []

    lines = ["Top Problem Areas (WUC):"]

    for i, (label, count) in enumerate(wuc_dict.items(), start=1):
        lines.append(f"{i}. {label:<60} {count}")

    lines.append("")
    return lines

def analyze_results(df, desc_map = None, **filters):
    """Run the full analysis on filtered records."""
    results = query_records(df, **filters)

    output = {
        "total_issues": len(results),
        "top_discrepancies": {},
        "top_fixes": {},
        "issues_by_month": {},
        "wuc_breakdown": {},
        "results": results,
        "filters": filters
    }

    if len(results) > 0:
        if "discrepancy_normalized" in results.columns:
            output["top_discrepancies"] = (
                results["discrepancy_normalized"]
                .astype(str)
                .value_counts()
                .head(3)
                .to_dict()
            )

        if "corrective_action_normalized" in results.columns:
            output["top_fixes"] = (
                results["corrective_action_normalized"]
                .astype(str)
                .value_counts()
                .head(3)
                .to_dict()
            )

        output["issues_by_month"] = issues_by_month(results)

        if "wuc" not in filters:
            output["wuc_breakdown"] = wuc_breakdown(results, desc_map)

    return output

'''def clean_wuc_desc(desc):
    desc = str(desc)

    # Remove "(SC)" and similar tags
    desc = re.sub(r"\(SC\)", "", desc)

    # Remove nested parentheses
    desc = re.sub(r"\(.*?\)", "", desc)

    # Collapse whitespace
    desc = " ".join(desc.split())

    return desc.strip()'''

def format_answer(analysis):
    """Format the analysis into readable text."""
    if analysis["total_issues"] == 0:
        return "No matching records found."

    lines = [f"Total issues: {analysis['total_issues']}", ""]

    if analysis["top_discrepancies"]:
        lines.append("Top discrepancies:")
        for issue, count in analysis["top_discrepancies"].items():
            lines.append(f"- {issue} ({count})")
        lines.append("")

    if analysis["top_fixes"]:
        lines.append("Top fixes:")
        for fix, count in analysis["top_fixes"].items():
            lines.append(f"- {fix} ({count})")
        lines.append("")

    if analysis["issues_by_month"]:
        lines.append("Issues by month:")

        for month, count in analysis["issues_by_month"].items():
            # Convert "2021-12" → datetime → "Dec 2021"
            formatted_month = pd.to_datetime(month).strftime("%b %Y")
            lines.append(f"- {formatted_month}: {count}")

        lines.append("")

    return "\n".join(lines)