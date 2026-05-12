"""Pluggable summarization adapters. NullAdapter is template-based and
enterprise-safe (no external calls, no hallucination). Additional adapters
(Claude, Gemma/Ollama, Vantage/GO81, etc.) can be added without touching
profile logic.

Adapters MAY optionally implement `summarize_stream(profile)` returning an
iterator of string chunks; the UI will progressively render the output if so.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator, Protocol


ANALYST_PROMPT = (
    "You are a senior KC-135 maintenance reliability analyst briefing a fleet "
    "manager. The PROFILE below is a JSON object describing one Work Unit Code. "
    "Write a flowing narrative report — 4-6 short paragraphs of connected prose, "
    "not a bulleted form — that reads like an experienced analyst talking through "
    "the data. Cover, in roughly this order and woven together naturally:\n"
    "  - what this WUC is (`wuc`, `description`) and its scale "
    "(`total_records`, `affected_tails`, `date_range`), with the single biggest "
    "takeaway up front;\n"
    "  - why it happens — the dominant failure mode from `top_discrepancy_phrases` "
    "/ `top_discrepancy_keywords` (quote a representative write-up verbatim with "
    "its count; if one mode dominates, give its share of the total) and the "
    "typical fix from `top_corrective_actions` / `top_corrective_keywords`;\n"
    "  - when — seasonality from `month_histogram` and the year-over-year trend "
    "from `year_histogram` (compare earliest vs. latest years with the numbers);\n"
    "  - where — the top bases from `base_distribution` and whether the problem "
    "is fleet-wide or concentrated;\n"
    "  - lifecycle and discovery — airframe-age skew from `flight_hour_buckets` "
    "and how these are caught from `when_discovered_phase` / `maint_type_phase`;\n"
    "  - related work from `cooccurring_wucs` and what it implies;\n"
    "  - close with 2-3 prioritized recommended actions, each tied to a specific "
    "number from the profile.\n\n"
    "Style: confident, readable prose. Use whatever the profile gives you and "
    "simply move past anything it doesn't — never write 'insufficient data', "
    "'not available', or call out gaps; just write about what's there. Use ONLY "
    "numbers present in the profile — never invent figures. No preamble, no "
    "restating these instructions, no closing meta-remarks.\n\n"
    "PROFILE:\n"
)


def _build_prompt(profile: dict[str, Any]) -> str:
    return ANALYST_PROMPT + json.dumps(profile, default=str, indent=2)


class SummaryAdapter(Protocol):
    name: str

    def available(self) -> bool: ...
    def summarize(self, profile: dict[str, Any]) -> str: ...


def _pct(part: int, total: int) -> str:
    if not total:
        return "0%"
    return f"{100.0 * part / total:.0f}%"


def _top_items(d: dict, n: int = 3) -> list[tuple[str, int]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]


class NullAdapter:
    """Deterministic template-based narrative. No LLM, no network."""

    name = "Template (offline)"

    def available(self) -> bool:
        return True

    def summarize(self, profile: dict[str, Any]) -> str:
        total = profile["total_records"]
        if total == 0:
            return f"No records found for WUC {profile['wuc']}."

        lines: list[str] = []
        wuc = profile["wuc"]
        desc = profile["description"]
        lines.append(f"**WUC {wuc} — {desc}**")
        lines.append(
            f"Based on {total:,} maintenance records across "
            f"{profile['affected_tails']} airframes"
            + (f" ({profile['date_range'][0]} to {profile['date_range'][1]})."
               if profile["date_range"] else ".")
        )
        lines.append("")

        # WHY
        if profile["top_discrepancy_keywords"]:
            kws = ", ".join(k for k, _ in profile["top_discrepancy_keywords"][:6])
            lines.append(f"**Why it occurs.** The most frequent discrepancy keywords are: {kws}.")
            if profile["top_discrepancy_phrases"]:
                top_phrase, top_count = profile["top_discrepancy_phrases"][0]
                lines.append(
                    f"The single most common write-up is "
                    f'"{top_phrase[:120]}" ({top_count} occurrences, '
                    f"{_pct(top_count, total)} of all reports)."
                )
        if profile["top_corrective_keywords"]:
            fixes = ", ".join(k for k, _ in profile["top_corrective_keywords"][:5])
            lines.append(f"Typical corrective actions involve: {fixes}.")
        lines.append("")

        # WHERE
        if profile["base_distribution"]:
            top_bases = _top_items(profile["base_distribution"], 3)
            base_str = "; ".join(
                f"{b} ({c}, {_pct(c, total)})" for b, c in top_bases
            )
            lines.append(f"**Where.** Top reporting bases: {base_str}.")
            lines.append("")

        # WHEN
        if profile["month_histogram"]:
            months = profile["month_histogram"]
            top_months = _top_items(months, 3)
            month_str = ", ".join(f"{m} ({c})" for m, c in top_months)
            lines.append(f"**When (seasonality).** Peak months: {month_str}.")
        if profile["year_histogram"]:
            years = profile["year_histogram"]
            trend = " -> ".join(f"{y}: {c}" for y, c in list(years.items())[-5:])
            lines.append(f"Year-over-year: {trend}.")
        lines.append("")

        # LIFECYCLE
        if profile["flight_hour_buckets"]:
            top_bucket = _top_items(profile["flight_hour_buckets"], 1)[0]
            lines.append(
                f"**Airframe lifecycle.** Most occurrences "
                f"({_pct(top_bucket[1], total)}) are on airframes in the "
                f"{top_bucket[0]} flight-hour band, suggesting this failure is "
                f"{'age-correlated' if 'High' in top_bucket[0] else 'not strongly age-driven'}."
            )
        if profile["when_discovered_phase"]:
            top_phase = _top_items(profile["when_discovered_phase"], 2)
            phase_str = "; ".join(f"{p} ({c}, {_pct(c, total)})" for p, c in top_phase)
            lines.append(f"**Discovery phase.** {phase_str}.")
        if profile["maint_type_phase"]:
            top_mt = _top_items(profile["maint_type_phase"], 2)
            mt_str = "; ".join(f"{p} ({c})" for p, c in top_mt)
            lines.append(f"Maintenance type: {mt_str}.")
        lines.append("")

        # CO-OCCURRENCE
        if profile["cooccurring_wucs"]:
            co = _top_items(profile["cooccurring_wucs"], 5)
            co_str = ", ".join(f"{w} ({c})" for w, c in co)
            lines.append(
                f"**Often opened alongside.** Other WUCs frequently worked on "
                f"the same job control number: {co_str}."
            )

        return "\n".join(lines)


class ClaudeAdapter:
    """Uses the Anthropic API if ANTHROPIC_API_KEY is set."""

    name = "Claude (Anthropic API)"

    def __init__(self, model: str = "claude-opus-4-6"):
        self.model = model

    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def summarize(self, profile: dict[str, Any]) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": _build_prompt(profile)}],
        )
        return resp.content[0].text


class GemmaAdapter:
    """Local Gemma 4 via Ollama. Offline, enterprise-safe, no API key.

    Default model is the EmbeddedGemma 4B variant (`gemma4:e4b`); swap to
    `gemma4:e2b` for lighter footprint or `gemma4:26b-a4b` for the MoE 26B
    (4B active) when GPU/RAM allows.
    """

    def __init__(self, model: str = "gemma4:e4b", num_ctx: int = 8192):
        self.model = model
        self.num_ctx = num_ctx
        self.name = f"Gemma 4 — {model} (local)"

    def available(self) -> bool:
        try:
            import ollama  # noqa: F401
        except ImportError:
            return False
        try:
            import ollama
            ollama.list()  # raises if daemon is not running
            return True
        except Exception:
            return False

    def summarize(self, profile: dict[str, Any]) -> str:
        return "".join(self.summarize_stream(profile))

    def summarize_stream(self, profile: dict[str, Any]) -> Iterator[str]:
        import ollama

        stream = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": _build_prompt(profile)}],
            stream=True,
            options={"temperature": 0.3, "num_ctx": self.num_ctx},
        )
        for chunk in stream:
            yield chunk["message"]["content"]


def available_adapters() -> list[SummaryAdapter]:
    candidates: list[SummaryAdapter] = [
        NullAdapter(),
        GemmaAdapter(),
        ClaudeAdapter(),
    ]
    return [a for a in candidates if a.available()]
