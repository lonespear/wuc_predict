"""Unified KC-135 maintenance analytics app.

Tab 1: Predict WUC from free text (wraps app.py logic).
Tab 2: Query/filter maintenance records (wraps sum_app.py logic).
Tab 3: WUC Profile summarizer (new).
"""
from __future__ import annotations

import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MaxNLocator

from data_config import resolve_data_path, resolve_lookup_path
from sum_utils import parse_user_query, analyze_results, format_answer
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
                    st.success(
                        f"**{top['wuc']}** — {top['system']} / {top['definition']}  \n"
                        f"Confidence: {top['confidence']:.1f}%"
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
                            "ℹ️ Corrective action was empty — model accuracy is "
                            "highest when both fields are provided. Top-3 above "
                            "may help if the leading prediction looks off."
                        )
                    st.info("Jump to the WUC Profile tab to see why, when, where, and lifecycle.")
                else:
                    st.error("No prediction returned.")
    except Exception as e:
        st.error(f"Model unavailable: {e}")

# ---------------------------------------------------------------- Tab 2
with tab_query:
    st.header("Ask a data-driven question")
    query = st.text_input(
        "Question:",
        placeholder="e.g. How many issues has 57-1508 had from Jan 2023 to Dec 2024?",
    )
    if st.button("Run Query", key="query_btn") and query.strip():
        filters = parse_user_query(query)
        analysis = analyze_results(df, desc_map=desc_map, **filters)
        st.subheader("Answer")
        st.text(format_answer(analysis))

        if analysis["issues_by_month"]:
            month_df = pd.DataFrame(
                list(analysis["issues_by_month"].items()), columns=["Month", "Count"]
            )
            month_df["Month"] = pd.to_datetime(month_df["Month"])
            month_df = month_df.sort_values("Month")
            month_df["Label"] = month_df["Month"].dt.strftime("%b %Y")
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(month_df["Label"], month_df["Count"])
                plt.xticks(rotation=45, ha="right")
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel("Month")
                ax.set_ylabel("Issues")
                plt.tight_layout()
                st.pyplot(fig)

        if analysis["wuc_breakdown"]:
            st.subheader("Top Problem Areas (WUC)")
            wuc_df = pd.DataFrame(
                list(analysis["wuc_breakdown"].items()), columns=["WUC", "Count"]
            )
            wuc_df.insert(0, "Rank", range(1, len(wuc_df) + 1))
            st.table(wuc_df)

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
