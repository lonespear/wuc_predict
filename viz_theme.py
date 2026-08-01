"""One registered Altair theme + chart helpers, so every chart in the app is
consistent and the fixes live in one place.

Design decisions and why
------------------------
* **Dark surface.** `.streamlit/config.toml` sets the app chrome; these values
  match it so charts sit on the same plane as the page instead of punching a
  white hole in it.

* **Count axes get `tickMinStep=1, format='d'`.** The co-occurrence chart was
  rendering ticks at 0.00, 0.05 … 3.00 — around sixty labels for a scale whose
  maximum value is 3. Records are integers; fractional ticks are meaningless.

* **Diverging blue↔red for concentration, neutral gray at 1.0.** Base
  concentration is *polarity* data — above or below what a base's overall
  maintenance volume predicts — so it takes a diverging scale with a neutral
  midpoint, not a categorical or sequential one. Raw-count bars were the
  problem: they rank the biggest bases first for every WUC and so contradicted
  a narrative that had already corrected for volume.

* **Sequential single-hue blue for the calendar heatmap.** Magnitude, one hue,
  light→dark. Never a rainbow.

Palette values are the validated reference instance; the diverging poles were
run through the six-check validator on both surfaces (worst-pair CVD ΔE 23.8
light / 25.7 dark against an ≥8 target; normal-vision 31.6 / 31.9 against a
≥15 floor; all contrast ≥3:1).
"""
from __future__ import annotations

import altair as alt

# --- Surfaces and ink (dark mode) -------------------------------------------
SURFACE = "#1a1a19"      # chart surface
PAGE = "#0d0d0d"         # page plane
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"    # axis + tick labels
GRID = "#2c2c2a"         # hairline gridline
BASELINE = "#383835"

# --- Encoding colors ---------------------------------------------------------
SERIES_1 = "#3987e5"     # dark-mode blue, categorical slot 1
DIVERGE_LOW = "#3987e5"  # below expected
DIVERGE_MID = "#383835"  # neutral gray — "as expected", must not read as a hue
DIVERGE_HIGH = "#d03b3b" # above expected
SEQUENTIAL = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _theme() -> dict:
    return {
        "config": {
            "background": SURFACE,
            "font": FONT,
            "title": {"color": INK_PRIMARY, "fontSize": 13, "fontWeight": 600,
                      "anchor": "start", "offset": 10},
            "axis": {
                "labelColor": INK_MUTED,
                "titleColor": INK_SECONDARY,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "titleFontWeight": 500,
                "gridColor": GRID,
                "gridWidth": 1,
                "domainColor": BASELINE,
                "tickColor": BASELINE,
                "labelPadding": 4,
            },
            "legend": {
                "labelColor": INK_SECONDARY,
                "titleColor": INK_SECONDARY,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "symbolType": "square",
            },
            "view": {"stroke": None, "continuousWidth": 400, "continuousHeight": 220},
            "bar": {"cornerRadiusEnd": 4},   # 4px rounded data-end, flat at baseline
            "range": {"heatmap": SEQUENTIAL, "ramp": SEQUENTIAL},
        }
    }


def register() -> None:
    """Register and enable the theme. Safe to call repeatedly."""
    try:  # Altair 5.5+ API
        alt.theme.register("wuc", enable=True)(_theme)
    except AttributeError:  # Altair <5.5
        alt.themes.register("wuc", _theme)
        alt.themes.enable("wuc")


def count_axis(title: str) -> alt.Axis:
    """Axis for integer record counts.

    Without tickMinStep a scale topping out at 3 gets ticks every 0.05 —
    ~60 labels, none of which can occur in the data.
    """
    return alt.Axis(title=title, format="d", tickMinStep=1, grid=True)


def concentration_chart(concentration: dict, height: int = 26):
    """Base concentration as deviation from expected — not raw counts.

    `index` is observed records / expected-from-that-base's-overall-volume.
    1.0 means the base is simply busy. The neutral midpoint is gray so
    "as expected" reads as nothing rather than as a third category.
    """
    if not concentration:
        return None
    rows = [
        {
            "Base": base,
            "index": vals["index"],
            "records": vals["records"],
            "expected": vals["expected_if_typical"],
        }
        for base, vals in concentration.items()
    ]
    rows.sort(key=lambda r: r["index"], reverse=True)
    hi = max(2.0, max(r["index"] for r in rows))

    return (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X("index:Q",
                    axis=alt.Axis(title="× expected for this base's volume",
                                  values=[0, 0.5, 1, 1.5, 2, 2.5, 3]),
                    scale=alt.Scale(domain=[0, hi])),
            y=alt.Y("Base:N", sort="-x", axis=alt.Axis(title=None, labelLimit=150)),
            color=alt.Color(
                "index:Q",
                scale=alt.Scale(domain=[0, 1, hi],
                                range=[DIVERGE_LOW, DIVERGE_MID, DIVERGE_HIGH]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Base:N"),
                alt.Tooltip("records:Q", title="Records"),
                alt.Tooltip("expected:Q", title="Expected if typical", format=".1f"),
                alt.Tooltip("index:Q", title="× expected", format=".2f"),
            ],
        )
        .properties(height=max(120, height * len(rows)))
    )
