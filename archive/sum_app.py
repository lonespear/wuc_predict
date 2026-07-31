import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sum_utils import parse_user_query, analyze_results, format_answer

from matplotlib.ticker import MaxNLocator

st.set_page_config(page_title="KC-135 Maintenance Query Tool", layout="wide")

st.title("KC-135 Maintenance Query Tool")
st.write("Ask a data-driven question about maintenance records.")

@st.cache_data
def load_data():
    df = pd.read_csv("FinalData.csv")
    return df

@st.cache_data
def load_wuc():
    df = pd.read_csv("kc135_wuc_lookup_levels.csv")
    return df

df = load_data()
lookup = load_wuc()
desc_map = dict(zip(lookup["wuc_code"], lookup["description"]))

query = st.text_input(
    "Enter your question:"
    #placeholder="Example: How many issues has 57-1508 had in the last 6 months?"
)

if st.button("Run Query"):
    if query.strip():
        filters = parse_user_query(query)
        analysis = analyze_results(df, desc_map=desc_map, **filters)
        answer = format_answer(analysis)

        st.subheader("Answer")
        st.text(answer)

        if analysis["issues_by_month"]:
            st.subheader("Issues by Month")

            month_df = pd.DataFrame(
                list(analysis["issues_by_month"].items()),
                columns=["Month", "Count"]
            )

            month_df["Month"] = pd.to_datetime(month_df["Month"])
            month_df = month_df.sort_values("Month")
            month_df["Label"] = month_df["Month"].dt.strftime("%b %Y")

            # 👇 Create centered narrow column
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))

                ax.bar(month_df["Label"], month_df["Count"])
                plt.xticks(rotation=45, ha='right')
                ax.tick_params(axis='x', labelsize=8)

                ax.set_xlabel("Month")
                ax.set_ylabel("Issues")
                
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                plt.tight_layout()

                st.pyplot(fig)

        if analysis["wuc_breakdown"]:
            st.subheader("Top Problem Areas (WUC)")

            wuc_df = pd.DataFrame(
                list(analysis["wuc_breakdown"].items()),
                columns=["WUC", "Count"]
            )

            # Add ranking
            wuc_df.insert(0, "Rank", range(1, len(wuc_df) + 1))

            st.table(wuc_df)

        #if not analysis["results"].empty:
            #st.subheader("Matching Records")
            #st.dataframe(analysis["results"], use_container_width=True)

        #st.subheader("Detected Filters")
        #st.json(filters)

    else:
        st.warning("Please enter a question.")