
"""BioDataLab — Scientific data analysis workspace."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from biodatalab.loader import validate_dataset
from biodatalab.cleaning import (
    clean_dataset,
    remove_empty_columns,
    remove_empty_rows,
    remove_duplicate_rows,
    fill_missing_numeric,
)
from biodatalab.statistics import (
    analyze_dataset,
    calculate_correlation,
)
from biodatalab.outliers import summarize_outliers
from biodatalab.report import generate_analysis_report
from biodatalab.export import export_report_json
from biodatalab.visualization import (
    plot_histogram,
    plot_scatter,
    plot_correlation_matrix,
)
from biodatalab.group_comparison import compare_groups
from biodatalab.nonparametric import (
    mann_whitney,
    wilcoxon_signed_rank,
    kruskal_wallis,
    spearman_correlation,
    holm_correction,
)
from biodatalab.anova import (
    one_way_anova,
    tukey_hsd,
    games_howell,
)
from biodatalab.regression import (
    fit_linear_regression,
    predict_linear_regression,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioDataLab — Research workspace",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL IDENTITY
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {
    --paper: #F4F2EA;
    --paper-dark: #EAE8DE;
    --forest: #183B32;
    --forest-light: #285448;
    --sage: #879A84;
    --sage-light: #DCE3D8;
    --ink: #252C27;
    --muted: #777E74;
    --line: #D8D8CC;
    --white: #FCFBF7;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 3rem;
    padding-bottom: 5rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

[data-testid="stSidebar"] {
    background: #EAE9DF;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
    padding-bottom: 25px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 25px;
}

.brand-symbol {
    width: 43px;
    height: 43px;
    background: var(--forest);
    color: #F4F2EA;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 25px;
}

.brand-name {
    color: var(--forest);
    font-family: 'Playfair Display', serif;
    font-size: 24px;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.sidebar-label {
    color: var(--sage) !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    margin-top: 25px;
    margin-bottom: 10px;
}

h1, h2, h3 {
    color: var(--forest) !important;
}

h1, h2 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 500 !important;
}

h1 {
    font-size: 44px !important;
}

h2 {
    font-size: 29px !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 17px !important;
}

.eyebrow {
    color: #82917D;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 10px;
    font-weight: 700;
    margin-bottom: 12px;
}

.page-heading {
    font-family: 'Playfair Display', serif;
    color: var(--forest);
    font-size: 43px;
    line-height: 1.12;
    margin-bottom: 10px;
}

.page-description {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    max-width: 700px;
}

.section-heading {
    font-family: 'Playfair Display', serif;
    color: var(--forest);
    font-size: 25px;
    margin-bottom: 5px;
}

.section-caption {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 18px;
}

.editorial-rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 27px 0;
}

.hero {
    background: var(--forest);
    padding: 52px;
    margin: 15px 0 35px;
    position: relative;
    overflow: hidden;
}

.hero-kicker {
    color: #B7C7B3;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 20px;
}

.hero-title {
    color: #F4F2EA;
    font-family: 'Playfair Display', serif;
    font-size: 49px;
    line-height: 1.1;
    max-width: 650px;
    margin-bottom: 18px;
}

.hero-text {
    color: #D1DCCF;
    font-size: 14px;
    line-height: 1.8;
    max-width: 540px;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    margin: 20px 0 35px;
}

.metric-cell {
    padding: 19px 15px 19px 0;
    border-right: 1px solid var(--line);
    margin-right: 15px;
}

.metric-cell:last-child {
    border-right: none;
}

.metric-label {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 9px;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--forest);
    font-family: 'Playfair Display', serif;
    font-size: 29px;
}

.feature-row {
    display: grid;
    grid-template-columns: 45px 1fr;
    gap: 17px;
    padding: 18px 0;
    border-bottom: 1px solid var(--line);
}

.feature-number {
    color: var(--sage);
    font-family: 'Playfair Display', serif;
    font-size: 22px;
}

.feature-title {
    color: var(--forest);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 5px;
}

.feature-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--forest);
    color: #F4F2EA;
    border: 1px solid var(--forest);
    border-radius: 0;
    padding: 10px 20px;
    font-size: 12px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--forest-light);
    color: white;
}

[data-baseweb="select"] > div {
    background: var(--white);
    border-color: var(--line);
    border-radius: 0;
}

.stFileUploader section {
    background: rgba(255,255,255,0.35);
    border: 1px dashed #AEB7A8;
    border-radius: 0;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 0;
}

[data-testid="stExpander"] {
    background: transparent;
    border: none;
    border-top: 1px solid var(--line);
    border-radius: 0;
}

[data-testid="stAlert"] {
    border-radius: 0;
}

.footer {
    border-top: 1px solid var(--line);
    margin-top: 60px;
    padding-top: 18px;
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    .hero {
        padding: 30px;
    }

    .hero-title {
        font-size: 35px;
    }

    .metric-strip {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# REUSABLE FUNCTIONS
# ============================================================

def page_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <div class="page-heading">{title}</div>
        <div class="page-description">{description}</div>
        <hr class="editorial-rule">
        """,
        unsafe_allow_html=True,
    )


def section_header(title, caption=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<div class="section-caption">{caption}</div>',
            unsafe_allow_html=True,
        )


def metric_strip(items):
    cells = ""

    for label, value in items:
        cells += f"""
        <div class="metric-cell">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    st.markdown(
        f'<div class="metric-strip">{cells}</div>',
        unsafe_allow_html=True,
    )


def footer():
    st.markdown("""
    <div class="footer">
        <span>BioDataLab · Scientific workspace</span>
        <span>Biological & Agricultural Data Analysis</span>
    </div>
    """, unsafe_allow_html=True)


def numeric_columns(data):
    return list(
        data.select_dtypes(include=[np.number]).columns
    )


def make_groups(data, value_column, group_column):
    working = data[[value_column, group_column]].copy()
    working[value_column] = pd.to_numeric(
        working[value_column],
        errors="coerce",
    )
    working = working.dropna()

    return {
        str(name): group[value_column].to_numpy(dtype=float)
        for name, group in working.groupby(group_column)
    }


def show_result(result):
    """Display a statistical result without losing its structure."""

    if "p_value" in result:
        p_value = result["p_value"]
        statistic = result.get("statistic", np.nan)

        col1, col2, col3 = st.columns(3)
        col1.metric("Test statistic", f"{statistic:.5g}")
        col2.metric("P-value", f"{p_value:.5g}")
        col3.metric(
            "Conclusion",
            "Significant" if p_value < result.get("alpha", 0.05)
            else "Not significant",
        )

    st.json(result)


def export_dataframe(data, filename="biodatalab_dataset.csv"):
    st.download_button(
        "Export CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-symbol">B</div>
        <div>
            <div class="brand-name">BioDataLab</div>
            <div class="brand-subtitle">Scientific workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-label">Workspace</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Dataset explorer",
            "Data cleaning",
            "Descriptive statistics",
            "Outlier analysis",
            "Group comparison",
            "Non-parametric tests",
            "ANOVA & post-hoc",
            "Regression",
            "Visualizations",
            "Scientific report",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-label">Data source</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Import a dataset",
        type=["csv", "xlsx", "xls"],
    )

    st.markdown("---")

    st.markdown("""
    <div style="color:#777E74;font-size:11px;line-height:1.7;">
    An open-source scientific environment for biological
    and agricultural data analysis.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA LOADING AND SESSION STATE
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.get("dataset_name")
        != uploaded_file.name
    ):

        try:
            if uploaded_file.name.lower().endswith(".csv"):
                loaded = pd.read_csv(uploaded_file)
            else:
                loaded = pd.read_excel(uploaded_file)

            if loaded.empty:
                st.error("The uploaded dataset is empty.")
            else:
                st.session_state["active_data"] = loaded.copy()
                st.session_state["original_data"] = loaded.copy()
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state.pop("last_regression", None)
                st.session_state.pop("generated_report", None)

        except Exception as error:
            st.error(f"Could not load dataset: {error}")

dataframe = st.session_state.get("active_data")

if dataframe is not None:
    dataframe = dataframe.copy()

    try:
        dataset_info = validate_dataset(dataframe)
    except Exception as error:
        st.error(f"Dataset validation failed: {error}")
        dataframe = None


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    if dataframe is None:

        st.markdown("""
        <div class="hero">
            <div class="hero-kicker">Research · Data · Discovery</div>
            <div class="hero-title">
                A scientific workspace<br>
                for biological data.
            </div>
            <div class="hero-text">
                Explore experimental data, perform statistical
                analyses and generate scientific reports
                in one research environment.
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_header(
            "A laboratory for your data",
            "A structured environment for biological and agricultural research.",
        )

        features = [
            ("01", "Data management", "Import, validate, inspect and clean datasets."),
            ("02", "Statistical inference", "Parametric tests, non-parametric tests and ANOVA."),
            ("03", "Regression", "Model relationships, assess coefficients and make predictions."),
            ("04", "Scientific reporting", "Generate and export structured analysis reports."),
        ]

        for number, title, description in features:
            st.markdown(
                f"""
                <div class="feature-row">
                    <div class="feature-number">{number}</div>
                    <div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-description">{description}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info("Import a CSV or Excel dataset from the sidebar to begin.")

    else:

        page_header(
            "Research workspace",
            "Your dataset.",
            "An overview of the experimental data currently loaded.",
        )

        metric_strip([
            ("Observations", f'{dataset_info["rows"]:,}'),
            ("Variables", f'{dataset_info["columns"]:,}'),
            ("Missing values", f'{dataset_info["missing_values"]:,}'),
            ("Duplicate rows", f'{dataset_info["duplicate_rows"]:,}'),
        ])

        section_header("Dataset preview", st.session_state.get("dataset_name", ""))

        st.dataframe(
            dataframe.head(12),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data structure**")
            st.write(f'{dataset_info["rows"]} observations')
            st.write(f'{dataset_info["columns"]} variables')

        with col2:
            st.markdown("**Data completeness**")
            completeness = (
                100 * (1 - dataset_info["missing_values"] / dataframe.size)
                if dataframe.size else 100
            )
            st.write(f"{completeness:.2f}% complete")
            st.write(f'{dataset_info["duplicate_rows"]} duplicate rows')


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset explorer":

    page_header(
        "01 · Data inspection",
        "Dataset explorer.",
        "Inspect observations, variable types and data completeness.",
    )

    if dataframe is None:
        st.info("Import a dataset to access the explorer.")

    else:

        metric_strip([
            ("Rows", f'{dataset_info["rows"]:,}'),
            ("Columns", f'{dataset_info["columns"]:,}'),
            ("Missing", f'{dataset_info["missing_values"]:,}'),
            ("Duplicates", f'{dataset_info["duplicate_rows"]:,}'),
        ])

        section_header("Observations")

        preview_size = st.selectbox(
            "Rows to display",
            [10, 25, 50, 100, 250],
            index=2,
        )

        st.dataframe(
            dataframe.head(preview_size),
            use_container_width=True,
            hide_index=True,
        )

        section_header(
            "Variable dictionary",
            "Data types and missing observations.",
        )

        variable_info = pd.DataFrame({
            "Variable": dataframe.columns.astype(str),
            "Data type": dataframe.dtypes.astype(str).values,
            "Missing values": dataframe.isna().sum().values,
            "Completeness (%)": (
                100 * (1 - dataframe.isna().mean())
            ).round(2).values,
        })

        st.dataframe(
            variable_info,
            use_container_width=True,
            hide_index=True,
        )

        export_dataframe(dataframe)


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data cleaning":

    page_header(
        "02 · Data preparation",
        "Data cleaning.",
        "Apply controlled cleaning operations to a working copy of your dataset.",
    )

    if dataframe is None:
        st.info("Import a dataset to begin cleaning.")

    else:

        section_header("Current data quality")

        metric_strip([
            ("Rows", len(dataframe)),
            ("Columns", len(dataframe.columns)),
            ("Missing cells", int(dataframe.isna().sum().sum())),
            ("Duplicate rows", int(dataframe.duplicated().sum())),
        ])

        st.warning(
            "Cleaning changes the active dataset in this session. "
            "The original uploaded dataset remains available for restoration."
        )

        with st.form("cleaning_form"):

            remove_blank_columns = st.checkbox(
                "Remove completely empty columns",
                value=True,
            )

            remove_blank_rows = st.checkbox(
                "Remove completely empty rows",
                value=True,
            )

            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=True,
            )

            fill_missing = st.checkbox(
                "Fill missing numerical values",
                value=False,
            )

            fill_strategy = st.selectbox(
                "Imputation strategy",
                ["median", "mean", "zero"],
            )

            submitted = st.form_submit_button("Apply cleaning")

        if submitted:

            cleaned = dataframe.copy()

            if remove_blank_columns:
                cleaned = remove_empty_columns(cleaned)

            if remove_blank_rows:
                cleaned = remove_empty_rows(cleaned)

            if remove_duplicates:
                cleaned = remove_duplicate_rows(cleaned)

            if fill_missing:
                cleaned = fill_missing_numeric(
                    cleaned,
                    strategy=fill_strategy,
                )

            st.session_state["active_data"] = cleaned
            st.session_state.pop("last_regression", None)

            st.success("Cleaning operations applied.")
            st.rerun()

        if st.button("Restore original uploaded dataset"):
            original = st.session_state.get("original_data")

            if original is not None:
                st.session_state["active_data"] = original.copy()
                st.session_state.pop("last_regression", None)
                st.rerun()

        section_header("Cleaned dataset preview")

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        export_dataframe(dataframe, "biodatalab_cleaned_dataset.csv")


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

elif page == "Descriptive statistics":

    page_header(
        "03 · Statistical exploration",
        "Descriptive statistics.",
        "Characterise the distributions and relationships between numeric variables.",
    )

    if dataframe is None:
        st.info("Import a dataset to access descriptive statistics.")

    else:

        statistics = analyze_dataset(dataframe)

        if not statistics:
            st.warning("No numeric columns were found.")

        else:

            stats_df = pd.DataFrame.from_dict(
                {
                    column: asdict(values)
                    for column, values in statistics.items()
                },
                orient="index",
            )

            stats_df.index.name = "Variable"

            section_header("Statistical summary")

            st.dataframe(
                stats_df,
                use_container_width=True,
            )

            section_header("Distribution analysis")

            selected_column = st.selectbox(
                "Select a numeric variable",
                list(statistics.keys()),
            )

            values = pd.to_numeric(
                dataframe[selected_column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if not values.empty:

                col1, col2, col3 = st.columns(3)

                col1.metric("Observations", len(values))
                col2.metric("Mean", f"{values.mean():.3f}")
                col3.metric("Standard deviation", f"{values.std():.3f}")

                fig, ax = plt.subplots(figsize=(10, 4))

                fig.patch.set_facecolor("#FCFBF7")
                ax.set_facecolor("#FCFBF7")

                ax.hist(
                    values,
                    bins="auto",
                    color="#879A84",
                    edgecolor="#183B32",
                )

                ax.set_xlabel(selected_column)
                ax.set_ylabel("Frequency")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y", alpha=0.15)

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            section_header("Correlation matrix")

            method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman", "kendall"],
            )

            if st.button("Calculate correlation matrix"):

                try:
                    correlation = calculate_correlation(
                        dataframe,
                        method=method,
                    )

                    st.dataframe(
                        correlation,
                        use_container_width=True,
                    )

                except Exception as error:
                    st.error(str(error))


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

elif page == "Outlier analysis":

    page_header(
        "04 · Data quality",
        "Outlier analysis.",
        "Identify observations that may require further scientific investigation.",
    )

    if dataframe is None:
        st.info("Import a dataset to begin.")

    else:

        method = st.selectbox(
            "Detection method",
            ["iqr", "zscore"],
            format_func=lambda x: (
                "Interquartile range (IQR)"
                if x == "iqr"
                else "Z-score"
            ),
        )

        if method == "iqr":
            multiplier = st.number_input(
                "IQR multiplier",
                min_value=0.1,
                value=1.5,
                step=0.1,
            )
            threshold = 3.0
        else:
            threshold = st.number_input(
                "Absolute Z-score threshold",
                min_value=0.1,
                value=3.0,
                step=0.1,
            )
            multiplier = 1.5

        if st.button("Run outlier analysis"):

            try:
                results = summarize_outliers(
                    dataframe,
                    method=method,
                    multiplier=multiplier,
                    threshold=threshold,
                )

                st.session_state["outlier_results"] = results

            except Exception as error:
                st.error(str(error))

        results = st.session_state.get("outlier_results")

        if results:

            total = sum(
                result["count"]
                for result in results.values()
            )

            metric_strip([
                ("Variables", len(results)),
                ("Detected observations", total),
                ("Method", method.upper()),
                ("Dataset rows", len(dataframe)),
            ])

            section_header(
                "Detection results",
                "An outlier is not necessarily an erroneous observation.",
            )

            for column, result in results.items():

                with st.expander(
                    f'{column} · {result["count"]} detected'
                ):

                    if result["count"]:

                        st.dataframe(
                            pd.DataFrame({
                                "Index": result["indices"],
                                "Value": result["values"],
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                    else:
                        st.write("No outliers detected.")


# ============================================================
# GROUP COMPARISON
# ============================================================

elif page == "Group comparison":

    page_header(
        "05 · Comparative statistics",
        "Group comparison.",
        "Compare a numerical response across independent experimental groups.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        numeric = numeric_columns(dataframe)

        if len(numeric) == 0:
            st.warning("No numeric variables are available.")

        else:

            value_column = st.selectbox(
                "Response variable",
                numeric,
                key="comparison_response",
            )

            group_candidates = [
                column for column in dataframe.columns
                if column != value_column
            ]

            if not group_candidates:
                st.warning("A grouping variable is required.")

            else:

                group_column = st.selectbox(
                    "Grouping variable",
                    group_candidates,
                    key="comparison_group",
                )

                confidence = st.slider(
                    "Confidence level",
                    min_value=0.80,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                )

                if st.button("Compare groups"):

                    try:

                        result = compare_groups(
                            dataframe,
                            value_column,
                            group_column,
                            confidence_level=confidence,
                        )

                        st.session_state["group_comparison_result"] = result

                    except Exception as error:
                        st.error(str(error))

                result = st.session_state.get("group_comparison_result")

                if result:

                    section_header("Group summaries")

                    summary = pd.DataFrame.from_dict(
                        result["groups"],
                        orient="index",
                    )

                    st.dataframe(summary, use_container_width=True)

                    section_header("Global statistical test")

                    st.write(result["test"]["name"])
                    st.metric(
                        "P-value",
                        f'{result["test"]["p_value"]:.6g}',
                    )

                    st.caption(
                        "The current implementation of the multi-group "
                        "test in compare_groups uses scipy.stats.f_oneway. "
                        "For heteroscedastic groups, use the dedicated "
                        "Welch ANOVA module below."
                    )

                    st.json(result["test"])

                    st.warning(
                        "This comparison assumes independent observations. "
                        "Repeated measurements from the same plant require "
                        "a repeated-measures or mixed-effects approach."
                    )


# ============================================================
# NON-PARAMETRIC TESTS
# ============================================================

elif page == "Non-parametric tests":

    page_header(
        "06 · Statistical inference",
        "Non-parametric methods.",
        "Rank-based tests and correlation methods for biological datasets.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        numeric = numeric_columns(dataframe)

        if not numeric:
            st.warning("No numeric variables are available.")

        else:

            test_name = st.selectbox(
                "Statistical method",
                [
                    "Mann–Whitney U",
                    "Wilcoxon signed-rank",
                    "Kruskal–Wallis",
                    "Spearman correlation",
                    "Holm correction",
                ],
            )

            alpha = st.select_slider(
                "Significance threshold",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05,
            )

            if test_name in [
                "Mann–Whitney U",
                "Kruskal–Wallis",
            ]:

                value_column = st.selectbox(
                    "Response variable",
                    numeric,
                )

                grouping_options = [
                    column for column in dataframe.columns
                    if column != value_column
                ]

                if grouping_options:

                    group_column = st.selectbox(
                        "Grouping variable",
                        grouping_options,
                    )

                    groups = make_groups(
                        dataframe,
                        value_column,
                        group_column,
                    )

                    st.write(
                        f"{len(groups)} groups found: "
                        + ", ".join(groups.keys())
                    )

                    if test_name == "Mann–Whitney U":

                        if len(groups) != 2:
                            st.warning(
                                "Mann–Whitney requires exactly two groups."
                            )

                        alternative = st.selectbox(
                            "Alternative hypothesis",
                            ["two-sided", "less", "greater"],
                        )

                        if st.button("Run Mann–Whitney test") and len(groups) == 2:

                            try:
                                result = mann_whitney(
                                    *list(groups.values()),
                                    alternative=alternative,
                                    alpha=alpha,
                                )
                                show_result(result)

                            except Exception as error:
                                st.error(str(error))

                    else:

                        if st.button("Run Kruskal–Wallis"):

                            try:
                                result = kruskal_wallis(
                                    groups,
                                    alpha=alpha,
                                )
                                show_result(result)

                            except Exception as error:
                                st.error(str(error))

                else:
                    st.warning("A grouping variable is required.")

            elif test_name == "Wilcoxon signed-rank":

                col1, col2 = st.columns(2)

                with col1:
                    before = st.selectbox(
                        "Before",
                        numeric,
                        key="wilcoxon_before",
                    )

                with col2:
                    after = st.selectbox(
                        "After",
                        numeric,
                        index=min(1, len(numeric) - 1),
                        key="wilcoxon_after",
                    )

                alternative = st.selectbox(
                    "Alternative hypothesis",
                    ["two-sided", "less", "greater"],
                    key="wilcoxon_alternative",
                )

                st.caption(
                    "Rows must represent the same experimental units "
                    "in both columns."
                )

                if st.button("Run Wilcoxon test"):

                    try:
                        result = wilcoxon_signed_rank(
                            dataframe[before],
                            dataframe[after],
                            alternative=alternative,
                            alpha=alpha,
                        )
                        show_result(result)

                    except Exception as error:
                        st.error(str(error))

            elif test_name == "Spearman correlation":

                col1, col2 = st.columns(2)

                with col1:
                    x_column = st.selectbox(
                        "Variable X",
                        numeric,
                    )

                with col2:
                    y_column = st.selectbox(
                        "Variable Y",
                        numeric,
                        index=min(1, len(numeric) - 1),
                    )

                if st.button("Calculate Spearman correlation"):

                    try:
                        result = spearman_correlation(
                            dataframe[x_column],
                            dataframe[y_column],
                            alpha=alpha,
                        )
                        show_result(result)

                    except Exception as error:
                        st.error(str(error))

            else:

                raw_values = st.text_input(
                    "P-values separated by commas",
                    value="0.001, 0.02, 0.5",
                )

                if st.button("Apply Holm correction"):

                    try:
                        p_values = [
                            float(value.strip())
                            for value in raw_values.split(",")
                        ]

                        result = holm_correction(
                            p_values,
                            alpha=alpha,
                        )

                        st.dataframe(
                            pd.DataFrame({
                                "Raw p-value": result["raw_p_values"],
                                "Adjusted p-value": result["adjusted_p_values"],
                                "Significant": result["significant"],
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception as error:
                        st.error(str(error))


# ============================================================
# ANOVA AND POST-HOC
# ============================================================

elif page == "ANOVA & post-hoc":

    page_header(
        "07 · Parametric inference",
        "Analysis of variance.",
        "Test differences between independent groups and perform post-hoc comparisons.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        numeric = numeric_columns(dataframe)

        if numeric:

            value_column = st.selectbox(
                "Response variable",
                numeric,
            )

            group_options = [
                column for column in dataframe.columns
                if column != value_column
            ]

            if group_options:

                group_column = st.selectbox(
                    "Grouping variable",
                    group_options,
                )

                groups = make_groups(
                    dataframe,
                    value_column,
                    group_column,
                )

                if len(groups) < 2:
                    st.warning("At least two groups are required.")

                else:

                    model_type = st.radio(
                        "ANOVA model",
                        ["Classical one-way ANOVA", "Welch ANOVA"],
                    )

                    alpha = st.select_slider(
                        "Significance threshold",
                        options=[0.01, 0.025, 0.05, 0.10],
                        value=0.05,
                        key="anova_alpha",
                    )

                    posthoc = st.selectbox(
                        "Post-hoc method",
                        [
                            "None",
                            "Tukey HSD",
                            "Games–Howell",
                        ],
                    )

                    if st.button("Run ANOVA"):

                        try:

                            result = one_way_anova(
                                groups,
                                alpha=alpha,
                                equal_variances=(
                                    model_type == "Classical one-way ANOVA"
                                ),
                            )

                            st.session_state["anova_result"] = result
                            st.session_state["anova_groups"] = groups
                            st.session_state["anova_posthoc"] = posthoc
                            st.session_state["anova_alpha_value"] = alpha

                        except Exception as error:
                            st.error(str(error))

                    result = st.session_state.get("anova_result")

                    if result:

                        section_header("ANOVA results")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "F statistic",
                            f'{result["statistic"]:.5g}',
                        )

                        col2.metric(
                            "P-value",
                            f'{result["p_value"]:.5g}',
                        )

                        col3.metric(
                            "Eta squared",
                            f'{result["effect_size_eta_squared"]:.4f}',
                        )

                        st.write(
                            "**Conclusion:**",
                            "Statistically significant difference."
                            if result["significant"]
                            else "No statistically significant difference detected.",
                        )

                        section_header("Group summaries")

                        st.dataframe(
                            pd.DataFrame.from_dict(
                                result["groups"],
                                orient="index",
                            ),
                            use_container_width=True,
                        )

                        section_header("Assumption diagnostics")

                        st.json(result["assumption_checks"])

                        if result["test"] == "Welch ANOVA":
                            st.caption(
                                "Welch ANOVA does not assume equal group variances."
                            )

                        if st.session_state.get("anova_posthoc") != "None":

                            section_header("Post-hoc comparisons")

                            selected_posthoc = st.session_state["anova_posthoc"]
                            stored_groups = st.session_state["anova_groups"]
                            stored_alpha = st.session_state["anova_alpha_value"]

                            try:

                                if selected_posthoc == "Tukey HSD":

                                    if result["test"] != "One-way ANOVA":
                                        st.warning(
                                            "Tukey HSD is designed for the "
                                            "equal-variance setting. Consider "
                                            "Games–Howell after Welch ANOVA."
                                        )
                                    else:
                                        posthoc_result = tukey_hsd(
                                            stored_groups,
                                            alpha=stored_alpha,
                                        )

                                        st.dataframe(
                                            pd.DataFrame(
                                                posthoc_result["comparisons"]
                                            ),
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                else:

                                    posthoc_result = games_howell(
                                        stored_groups,
                                        alpha=stored_alpha,
                                    )

                                    st.dataframe(
                                        pd.DataFrame(
                                            posthoc_result["comparisons"]
                                        ),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                            except Exception as error:
                                st.error(str(error))

                        st.warning(
                            "ANOVA requires independent observations. "
                            "Repeated observations from the same plants "
                            "must not be treated as independent replicates."
                        )

            else:
                st.warning("A grouping variable is required.")

        else:
            st.warning("No numeric variables are available.")


# ============================================================
# REGRESSION
# ============================================================

elif page == "Regression":

    page_header(
        "08 · Predictive modelling",
        "Linear regression.",
        "Estimate relationships between biological variables and evaluate model performance.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        numeric = numeric_columns(dataframe)

        if len(numeric) < 2:
            st.warning("At least two numeric variables are required.")

        else:

            outcome = st.selectbox(
                "Dependent variable (Y)",
                numeric,
            )

            predictor_options = [
                column for column in numeric
                if column != outcome
            ]

            predictors = st.multiselect(
                "Independent variables (X)",
                predictor_options,
                default=[predictor_options[0]],
            )

            alpha = st.select_slider(
                "Significance threshold",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05,
                key="regression_alpha",
            )

            if st.button("Fit regression model"):

                try:

                    if not predictors:
                        raise ValueError(
                            "Select at least one predictor."
                        )

                    result = fit_linear_regression(
                        dataframe,
                        outcome,
                        predictors,
                        alpha=alpha,
                    )

                    st.session_state["last_regression"] = result

                except Exception as error:
                    st.error(str(error))

            result = st.session_state.get("last_regression")

            if result:

                section_header("Model performance")

                model = result["model"]

                metric_strip([
                    ("R²", f'{model["r_squared"]:.4f}'),
                    ("Adjusted R²", f'{model["adjusted_r_squared"]:.4f}'),
                    ("Observations", result["n_observations"]),
                    ("Dropped rows", result["dropped_rows"]),
                ])

                st.write(
                    "**Global model p-value:**",
                    f'{model["p_value"]:.6g}',
                )

                section_header("Regression coefficients")

                coefficients = pd.DataFrame.from_dict(
                    result["coefficients"],
                    orient="index",
                )

                coefficients.index.name = "Parameter"

                st.dataframe(
                    coefficients,
                    use_container_width=True,
                )

                section_header("Model diagnostics")

                st.json(result["diagnostics"])

                st.warning(
                    "The usual OLS inferential statistics assume independent "
                    "observations and appropriate residual behaviour. "
                    "Repeated measurements from the same plant require "
                    "a longitudinal or mixed-effects model."
                )

                section_header("Prediction")

                st.caption(
                    "Enter predictor values to calculate a prediction "
                    "from the fitted model."
                )

                with st.form("prediction_form"):

                    prediction_values = {}

                    for predictor in result["predictors"]:

                        prediction_values[predictor] = st.number_input(
                            predictor,
                            value=float(
                                pd.to_numeric(
                                    dataframe[predictor],
                                    errors="coerce",
                                ).median()
                            ),
                            format="%.6f",
                        )

                    predict_button = st.form_submit_button(
                        "Calculate prediction"
                    )

                if predict_button:

                    try:

                        prediction_data = pd.DataFrame([
                            prediction_values
                        ])

                        prediction = predict_linear_regression(
                            prediction_data,
                            result["predictors"],
                            result["coefficients"],
                        )

                        st.metric(
                            f"Predicted {result['outcome']}",
                            f"{prediction[0]:.6g}",
                        )

                    except Exception as error:
                        st.error(str(error))


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "Visualizations":

    page_header(
        "09 · Scientific visualization",
        "Visual exploration.",
        "Create publication-oriented plots from your experimental dataset.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        numeric = numeric_columns(dataframe)

        if numeric:

            plot_type = st.selectbox(
                "Visualization",
                [
                    "Histogram",
                    "Scatter plot",
                    "Correlation matrix",
                ],
            )

            if plot_type == "Histogram":

                column = st.selectbox(
                    "Variable",
                    numeric,
                )

                bins = st.slider(
                    "Number of bins",
                    min_value=5,
                    max_value=60,
                    value=20,
                )

                if st.button("Generate histogram"):

                    temporary_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            suffix=".png",
                            delete=False,
                        ) as temporary:
                            temporary_path = Path(temporary.name)

                        plot_histogram(
                            dataframe,
                            column,
                            temporary_path,
                            bins=bins,
                        )

                        st.image(
                            str(temporary_path),
                            use_container_width=True,
                        )

                    except Exception as error:
                        st.error(str(error))

                    finally:
                        if temporary_path and temporary_path.exists():
                            temporary_path.unlink()

            elif plot_type == "Scatter plot":

                col1, col2 = st.columns(2)

                with col1:
                    x_column = st.selectbox("X variable", numeric)

                with col2:
                    y_column = st.selectbox(
                        "Y variable",
                        numeric,
                        index=min(1, len(numeric) - 1),
                    )

                if st.button("Generate scatter plot"):

                    temporary_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            suffix=".png",
                            delete=False,
                        ) as temporary:
                            temporary_path = Path(temporary.name)

                        plot_scatter(
                            dataframe,
                            x_column,
                            y_column,
                            temporary_path,
                        )

                        st.image(
                            str(temporary_path),
                            use_container_width=True,
                        )

                    except Exception as error:
                        st.error(str(error))

                    finally:
                        if temporary_path and temporary_path.exists():
                            temporary_path.unlink()

            else:

                method = st.selectbox(
                    "Correlation method",
                    ["pearson", "spearman", "kendall"],
                    key="visual_correlation_method",
                )

                if st.button("Generate correlation matrix"):

                    temporary_path = None

                    try:

                        with tempfile.NamedTemporaryFile(
                            suffix=".png",
                            delete=False,
                        ) as temporary:
                            temporary_path = Path(temporary.name)

                        plot_correlation_matrix(
                            dataframe,
                            temporary_path,
                            method=method,
                        )

                        st.image(
                            str(temporary_path),
                            use_container_width=True,
                        )

                    except Exception as error:
                        st.error(str(error))

                    finally:
                        if temporary_path and temporary_path.exists():
                            temporary_path.unlink()

        else:
            st.warning("No numeric variables are available.")


# ============================================================
# SCIENTIFIC REPORT
# ============================================================

elif page == "Scientific report":

    page_header(
        "10 · Documentation",
        "Scientific report.",
        "Generate a structured summary of your dataset and its statistical characteristics.",
    )

    if dataframe is None:
        st.info("Import a dataset first.")

    else:

        method = st.selectbox(
            "Outlier detection method",
            ["iqr", "zscore"],
        )

        if st.button("Generate report"):

            try:

                report = generate_analysis_report(
                    dataframe,
                    outlier_method=method,
                )

                st.session_state["generated_report"] = report

            except Exception as error:
                st.error(str(error))

        report = st.session_state.get("generated_report")

        if report:

            section_header("Report preview")

            st.json(report)

            try:

                with tempfile.NamedTemporaryFile(
                    suffix=".json",
                    delete=False,
                ) as temporary:
                    report_path = Path(temporary.name)

                export_report_json(
                    report,
                    report_path,
                )

                report_bytes = report_path.read_bytes()
                report_path.unlink()

                st.download_button(
                    "Download scientific report",
                    data=report_bytes,
                    file_name="biodatalab_report.json",
                    mime="application/json",
                )

            except Exception as error:
                st.error(f"Report export failed: {error}")


# ============================================================
# FOOTER
# ============================================================

footer()
