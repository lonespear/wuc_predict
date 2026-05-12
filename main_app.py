"""Unified KC-135 maintenance analytics app.

Tab 1: Predict WUC from free text (wraps app.py logic).
Tab 2: Query/filter maintenance records (wraps sum_app.py logic).
Tab 3: WUC Profile summarizer (new).
"""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from data_config import resolve_data_path, resolve_lookup_path
from sum_utils import parse_user_query, analyze_results, describe_filters
from wuc_profile import build_profile
from llm_adapter import available_adapters

st.set_page_config(page_title="KC-135 Maintenance Analytics", page_icon="✈️", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(resolve_data_path(), low_memory=False)


@st.cache_data
def load_desc_map() -> dict[str, str]:
    lookup_path = resolve_lookup_path()
    if lookup_path is not None:
        lookup = pd.read_csv(lookup_path)
        code_col = "wuc_code" if "wuc_code" in lookup.columns else lookup.columns[0]
        desc_col = "description" if "description" in lookup.columns else lookup.columns[1]
        return dict(zip(lookup[code_col].astype(str), lookup[desc_col].astype(str)))
    try:
        with open("codes.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


st.title("✈️ KC-135 Maintenance Analytics")

try:
    df = load_data()
    desc_map = load_desc_map()
    st.caption(f"Loaded {len(df):,} records from `{resolve_data_path().name}`")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_predict, tab_query, tab_profile = st.tabs(
    ["🔮 Predict WUC", "🔎 Query Records", "📊 WUC Profile"]
)

# ---------------------------------------------------------------- Tab 1
with tab_predict:
    st.header("Predict Work Unit Code from free text")
    st.caption(
        "Provide the discrepancy and (when available) the corrective action — "
        "the model was trained on both fields joined together."
    )

    @st.cache_resource
    def get_predictor():
        from model_loader import predict_top_k, build_input_text
        return predict_top_k, build_input_text

    try:
        predict_top_k, build_input_text = get_predictor()
        col_d, col_c = st.columns(2)
        with col_d:
            discrepancy = st.text_area(
                "Discrepancy",
                height=140,
                placeholder="e.g. PILOT SEAT REQUIRES EXCESSIVE FORCE TO ADJUST",
            )
        with col_c:
            corrective = st.text_area(
                "Corrective Action (optional but improves accuracy)",
                height=140,
                placeholder="e.g. REPLACED LATERAL SEAT ADJUSTER PER TM 1C-135-06",
            )

        if st.button("Predict WUC", key="predict_btn"):
            if not discrepancy.strip():
                st.warning("Discrepancy text is required.")
            else:
                text = build_input_text(discrepancy, corrective)
                results = predict_top_k(text, k=3)
                if results:
                    top = results[0]
                    conf = top["confidence"]
                    header = f"**{top['wuc']}** — {top['system']} / {top['definition']}  \nConfidence: {conf:.1f}%"

                    # Confidence bands — be honest with the user about uncertainty
                    if conf >= 70:
                        st.success(header)
                    elif conf >= 30:
                        st.warning(header + "  \n_Moderate confidence — review the alternatives below._")
                    else:
                        st.error(
                            header
                            + "  \n_⚠️ Low confidence — the model is uncertain about this input._"
                            + "  \nLikely causes: input is out-of-distribution (informal phrasing, "
                            "uncommon issue, missing corrective action). Treat the top-1 as a guess "
                            "and review all candidates below."
                        )
                    st.session_state["predicted_wuc"] = top["wuc"]

                    if len(results) > 1:
                        st.markdown("**Other candidates**")
                        for r in results[1:]:
                            st.markdown(
                                f"- `{r['wuc']}` — {r['system']} / {r['definition']} "
                                f"({r['confidence']:.1f}%)"
                            )

                    if not corrective.strip():
                        st.caption(
                            "ℹ️ Corrective action was empty — the model was trained on "
                            "discrepancy + corrective action joined together. Accuracy is "
                            "much higher when both fields are provided. The model also expects "
                            "maintenance-report style (all caps, terse, technical), e.g. "
                            "`PILOT SEAT BELT FRAYED` rather than `seatbelt is frayed`."
                        )
                    st.info("Jump to the WUC Profile tab to see why, when, where, and lifecycle.")
                else:
                    st.error("No prediction returned.")
    except Exception as e:
        st.error(f"Model unavailable: {e}")

# ---------------------------------------------------------------- Tab 2
def _hbar(data: dict, value_name: str, label_name: str, label_limit: int = 60):
    """Horizontal bar chart from a {label: count} dict, biggest on top."""
    import altair as alt

    d = pd.DataFrame(list(data.items()), columns=[label_name, value_name])
    d[label_name] = d[label_name].astype(str).str.slice(0, label_limit)
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"{value_name}:Q", title=value_name),
            y=alt.Y(f"{label_name}:N", sort="-x", title=None),
            tooltip=[label_name, value_name],
        )
        .properties(height=max(120, 26 * len(d)))
    )


with tab_query:
    st.header("Ask a data-driven question")
    query = st.text_input(
        "Question:",
        placeholder="e.g. How many 14AD0 issues since summer 2025? · issues for 57-1508 from Jan 2023 to Dec 2024",
    )
    run = st.button("Run Query", key="query_btn")

    if run and query.strip():
        filters = parse_user_query(query)
        analysis = analyze_results(df, desc_map=desc_map, **filters)

        # --- 1. Echo the interpreted filters so the user trusts the answer ---
        applied = describe_filters(filters)
        if applied:
            st.info("🔎 Interpreted as: " + " · ".join(applied))
        else:
            st.warning(
                "⚠️ No tail / WUC / date filter was recognized in that question — "
                "showing **all records**. Try e.g. `WUC 14AD0 since summer 2025` "
                "or `57-1508 from Jan 2023 to Dec 2024`."
            )

        total = analysis["total_issues"]
        if total == 0:
            st.error("No matching records found.")
        else:
            # --- 2. Headline metrics ------------------------------------
            pct = 100.0 * total / len(df)
            cols = st.columns(5)
            cols[0].metric("Records", f"{total:,}", help=f"{pct:.1f}% of all {len(df):,} records")
            cols[1].metric("Distinct airframes", f"{analysis['distinct_tails']:,}")
            cols[2].metric("Distinct WUCs", f"{analysis['distinct_wucs']:,}")
            if analysis["date_span"]:
                cols[3].metric("Covers", f"{analysis['date_span'][0]} → {analysis['date_span'][1]}")
            prior = analysis["prior_period_count"]
            if prior is not None:
                delta = total - prior
                pct_delta = (100.0 * delta / prior) if prior else 0.0
                cols[4].metric("vs prior period", f"{prior:,}", delta=f"{delta:+,} ({pct_delta:+.0f}%)")

            # --- 3. Time series (line, not 70 rotated bars) -------------
            if analysis["issues_by_month"]:
                st.subheader("Volume over time")
                m = pd.DataFrame(analysis["issues_by_month"].items(), columns=["Month", "Issues"])
                m["Month"] = pd.to_datetime(m["Month"])
                m = m.sort_values("Month").set_index("Month")
                if len(m) >= 6:
                    m["12-mo avg"] = m["Issues"].rolling(12, min_periods=3).mean()
                st.line_chart(m)

            # --- 4. Top discrepancies / fixes side by side -------------
            c_disc, c_fix = st.columns(2)
            with c_disc:
                if analysis["top_discrepancies"]:
                    st.subheader("Top discrepancies")
                    st.altair_chart(_hbar(analysis["top_discrepancies"], "Count", "Discrepancy"), use_container_width=True)
                    with st.expander("raw counts"):
                        st.dataframe(pd.DataFrame(analysis["top_discrepancies"].items(), columns=["Discrepancy", "Count"]), hide_index=True, use_container_width=True)
            with c_fix:
                if analysis["top_fixes"]:
                    st.subheader("Top corrective actions")
                    st.altair_chart(_hbar(analysis["top_fixes"], "Count", "Corrective action"), use_container_width=True)
                    with st.expander("raw counts"):
                        st.dataframe(pd.DataFrame(analysis["top_fixes"].items(), columns=["Corrective action", "Count"]), hide_index=True, use_container_width=True)

            # --- 5. Where + seasonality --------------------------------
            c_base, c_seas = st.columns(2)
            with c_base:
                if analysis["base_breakdown"]:
                    st.subheader("By base")
                    st.altair_chart(_hbar(analysis["base_breakdown"], "Count", "Base"), use_container_width=True)
            with c_seas:
                if analysis["seasonality"] and sum(analysis["seasonality"].values()) > 0:
                    st.subheader("Seasonality (month-of-year, all years)")
                    s = pd.DataFrame(analysis["seasonality"].items(), columns=["Month", "Issues"]).set_index("Month")
                    st.bar_chart(s)

            # --- 6. Top problem-area WUCs (drill-through) --------------
            if analysis["wuc_breakdown"]:
                st.subheader("Top problem areas (WUC)")
                st.altair_chart(_hbar(analysis["wuc_breakdown"], "Count", "WUC", label_limit=70), use_container_width=True)
                wuc_codes = [lbl.split(" ")[0] for lbl in analysis["wuc_breakdown"]]
                pick = st.selectbox("Inspect a WUC in the Profile tab:", ["—"] + wuc_codes, key="q_wuc_pick")
                if pick != "—":
                    st.session_state["predicted_wuc"] = pick
                    st.success(f"Set **{pick}** — switch to the 📊 WUC Profile tab and click *Build Profile*.")

            # --- 7. Worst airframes (only when not already tail-filtered)
            if analysis["tail_leaderboard"]:
                with st.expander("Worst airframes for this query (top 10 by record count)"):
                    st.dataframe(
                        pd.DataFrame(analysis["tail_leaderboard"].items(), columns=["Tail Number", "Records"]),
                        hide_index=True, use_container_width=True,
                    )

            # --- 8. Underlying records ---------------------------------
            with st.expander(f"Show matching records ({total:,} rows)"):
                show = analysis["results"]
                st.dataframe(show.head(2000), use_container_width=True, hide_index=True)
                if len(show) > 2000:
                    st.caption("Showing first 2,000 rows; download for the full set.")
                st.download_button(
                    "Download all matching records (CSV)",
                    show.to_csv(index=False).encode("utf-8"),
                    file_name="wuc_query_results.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------- Tab 3
with tab_profile:
    st.header("WUC Profile — why, when, where, lifecycle")

    adapters = available_adapters()
    adapter_names = [a.name for a in adapters]

    col_input, col_adapter = st.columns([2, 1])
    with col_input:
        default_wuc = st.session_state.get("predicted_wuc", "12AA0")
        wuc_input = st.text_input("WUC code:", value=default_wuc).strip().upper()
    with col_adapter:
        adapter_choice = st.selectbox("Summary engine:", adapter_names)

    if st.button("Build Profile", key="profile_btn") and wuc_input:
        with st.spinner("Analyzing records..."):
            profile = build_profile(df, wuc_input, desc_map=desc_map)

        if profile["total_records"] == 0:
            st.warning(f"No records found for WUC {wuc_input}.")
        else:
            adapter = adapters[adapter_names.index(adapter_choice)]

            st.subheader("Narrative Summary")
            placeholder = st.empty()
            if hasattr(adapter, "summarize_stream"):
                narrative = ""
                for chunk in adapter.summarize_stream(profile):
                    narrative += chunk
                    placeholder.markdown(narrative)
            else:
                narrative = adapter.summarize(profile)
                placeholder.markdown(narrative)

            st.divider()
            st.subheader("Supporting Breakdowns")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total records", f"{profile['total_records']:,}")
            m2.metric("Affected airframes", profile["affected_tails"])
            if profile["date_range"]:
                m3.metric("Date range", f"{profile['date_range'][0][:7]} → {profile['date_range'][1][:7]}")

            c_when, c_where = st.columns(2)
            with c_when:
                if profile["month_histogram"]:
                    st.markdown("**Seasonality (by month)**")
                    mh = pd.DataFrame(
                        profile["month_histogram"].items(), columns=["Month", "Count"]
                    )
                    st.bar_chart(mh.set_index("Month"))
                if profile["year_histogram"]:
                    st.markdown("**Year-over-year**")
                    yh = pd.DataFrame(
                        profile["year_histogram"].items(), columns=["Year", "Count"]
                    )
                    st.bar_chart(yh.set_index("Year"))
            with c_where:
                if profile["base_distribution"]:
                    st.markdown("**By base**")
                    bd = pd.DataFrame(
                        profile["base_distribution"].items(), columns=["Base", "Count"]
                    )
                    st.bar_chart(bd.set_index("Base"))

            c_life, c_phase = st.columns(2)
            with c_life:
                if profile["flight_hour_buckets"]:
                    st.markdown("**Airframe lifecycle (flight-hour quartiles)**")
                    fh = pd.DataFrame(
                        profile["flight_hour_buckets"].items(), columns=["Band", "Count"]
                    )
                    st.table(fh)
            with c_phase:
                if profile["when_discovered_phase"]:
                    st.markdown("**Discovery phase**")
                    wd = pd.DataFrame(
                        profile["when_discovered_phase"].items(), columns=["Phase", "Count"]
                    )
                    st.table(wd)

            c_disc, c_fix = st.columns(2)
            with c_disc:
                if profile["top_discrepancy_phrases"]:
                    st.markdown("**Top discrepancy write-ups**")
                    for phrase, count in profile["top_discrepancy_phrases"]:
                        st.markdown(f"- *{phrase[:200]}* ({count})")
            with c_fix:
                if profile["top_corrective_actions"]:
                    st.markdown("**Top corrective actions**")
                    for phrase, count in profile["top_corrective_actions"]:
                        st.markdown(f"- *{phrase[:200]}* ({count})")

            if profile["cooccurring_wucs"]:
                st.markdown("**Frequently opened alongside (same JCN)**")
                co = pd.DataFrame(
                    profile["cooccurring_wucs"].items(), columns=["WUC", "Count"]
                )
                st.table(co)

            with st.expander("Raw profile JSON (for debugging / export)"):
                st.json(profile, expanded=False)
