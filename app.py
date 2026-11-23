
"""BioDataLab - Interactive scientific analysis application."""

import json

import pandas as pd
import streamlit as st

from biodatalab.loader import validate_dataset
from biodatalab.statistics import analyze_dataset
from biodatalab.outliers import summarize_outliers
from biodatalab.report import generate_analysis_report


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="BioDataLab",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 BioDataLab")
st.caption(
    "Biological and Agricultural Data Analysis Platform"
)

st.sidebar.title("Analysis workspace")

uploaded_file = st.sidebar.file_uploader(
    "Import a biological dataset",
    type=["csv", "xlsx", "xls"],
)


# --------------------------------------------------
# Welcome screen
# --------------------------------------------------

if uploaded_file is None:
    st.info(
        "Upload a CSV or Excel file to begin your analysis."
    )

    st.markdown("""
    ### Welcome to BioDataLab

    Explore your experimental datasets with:

    - Dataset validation
    - Descriptive statistics
    - Outlier detection
    - Scientific analysis reports
    """)

    st.stop()


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

try:
    if uploaded_file.name.lower().endswith(".csv"):
        dataframe = pd.read_csv(uploaded_file)
    else:
        dataframe = pd.read_excel(uploaded_file)

    if dataframe.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

except Exception as error:
    st.error(f"Could not read the uploaded file: {error}")
    st.stop()


st.success(f"Dataset loaded: {uploaded_file.name}")


# --------------------------------------------------
# Navigation tabs
# --------------------------------------------------

tab_data, tab_stats, tab_outliers, tab_report = st.tabs([
    "Dataset",
    "Statistics",
    "Outliers",
    "Scientific report",
])


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

with tab_data:
    st.subheader("Dataset overview")

    info = validate_dataset(dataframe)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", info["rows"])
    col2.metric("Columns", info["columns"])
    col3.metric("Missing values", info["missing_values"])
    col4.metric("Duplicate rows", info["duplicate_rows"])

    st.subheader("Data preview")

    st.dataframe(
        dataframe.head(100),
        use_container_width=True,
    )

    st.subheader("Column information")

    column_info = pd.DataFrame({
        "Column": dataframe.columns.astype(str),
        "Data type": dataframe.dtypes.astype(str).values,
        "Missing values": dataframe.isna().sum().values,
    })

    st.dataframe(
        column_info,
        use_container_width=True,
    )


# --------------------------------------------------
# Descriptive statistics
# --------------------------------------------------

with tab_stats:
    st.subheader("Descriptive statistics")

    statistics = analyze_dataset(dataframe)

    if not statistics:
        st.warning("No numeric columns were found.")

    else:
        stats_df = pd.DataFrame.from_dict(
            statistics,
            orient="index",
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
        )

        selected_column = st.selectbox(
            "Choose a numeric variable",
            list(statistics.keys()),
        )

        values = pd.to_numeric(
            dataframe[selected_column],
            errors="coerce",
        ).dropna()

        st.subheader("Value distribution")

        st.bar_chart(
            values.value_counts().sort_index()
        )


# --------------------------------------------------
# Outlier detection
# --------------------------------------------------

with tab_outliers:
    st.subheader("Outlier detection")

    method = st.selectbox(
        "Detection method",
        ["iqr", "zscore"],
    )

    if st.button("Run outlier detection"):

        results = summarize_outliers(
            dataframe,
            method=method,
        )

        if not results:
            st.info("No numeric columns available.")

        else:
            for column, result in results.items():

                with st.expander(column):

                    st.metric(
                        "Detected outliers",
                        result["count"],
                    )

                    if result["count"]:
                        st.write(
                            "Indices:",
                            result["indices"],
                        )

                        st.write(
                            "Values:",
                            result["values"],
                        )


# --------------------------------------------------
# Scientific report
# --------------------------------------------------

with tab_report:
    st.subheader("Scientific analysis report")

    report_method = st.selectbox(
        "Outlier detection method",
        ["iqr", "zscore"],
        key="report_method",
    )

    if st.button("Generate report"):

        try:
            report = generate_analysis_report(
                dataframe,
                outlier_method=report_method,
            )

            st.json(report)

            report_bytes = json.dumps(
                report,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")

            st.download_button(
                "Download JSON report",
                data=report_bytes,
                file_name="biodatalab_report.json",
                mime="application/json",
            )

        except Exception as error:
            st.error(
                f"Report generation failed: {error}"
            )
