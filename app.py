from __future__ import annotations

import io
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioDataLab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 BioDataLab")
st.caption(
    "Open-source scientific, biological and agricultural "
    "data analysis platform."
)

# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "original_data" not in st.session_state:
    st.session_state.original_data = None

if "results" not in st.session_state:
    st.session_state.results = {}

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# UTILITIES
# ============================================================

def record_result(name, result):
    st.session_state.results[name] = result

    st.session_state.history.append(
        {
            "analysis": name,
            "timestamp": datetime.now().isoformat(),
        }
    )


def get_data():
    return st.session_state.data


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def display_dataframe(df):
    st.dataframe(df, use_container_width=True)


def download_csv(df, filename="biodatalab_results.csv"):
    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def download_json(data, filename="biodatalab_results.json"):
    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(data, indent=2, default=str),
        file_name=filename,
        mime="application/json",
    )


def safe_numeric_df(df):
    return df.select_dtypes(include=np.number).copy()


def show_error(error):
    st.error(str(error))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("BioDataLab")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📂 Data",
        "🧹 Data cleaning",
        "📊 Descriptive statistics",
        "🧪 Statistical tests",
        "📈 Regression",
        "🔗 Correlation",
        "🎯 Sampling",
        "📉 Outliers",
        "🧬 PCA",
        "🔬 Clustering",
        "📅 Time series",
        "⚡ Bootstrap",
        "📐 Effect size",
        "🔋 Statistical power",
        "📋 Data profiling",
        "📑 Reports",
        "📤 Export",
    ],
)

# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(uploaded_file):

    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    if filename.endswith(".json"):
        return pd.read_json(uploaded_file)

    if filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)

    raise ValueError("Unsupported file format.")


# ============================================================
# OVERVIEW
# ============================================================

if navigation == "🏠 Overview":

    st.header("Scientific data analysis")

    st.markdown(
        """
        BioDataLab is a modular Python toolkit designed to facilitate
        biological, agricultural and scientific data analysis.

        Its objective is to provide accessible tools for:

        - Data preparation and quality control
        - Statistical analysis
        - Sampling and representativeness
        - Regression and predictive analysis
        - Dimensionality reduction
        - Clustering
        - Visualization and reporting
        """
    )

    st.divider()

    df = get_data()

    if df is not None:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Numeric variables", len(numeric_columns(df)))
        col4.metric("Missing values", int(df.isna().sum().sum()))

        st.subheader("Current dataset")
        display_dataframe(df.head(10))

    else:

        st.info(
            "Upload a dataset from the Data section "
            "to begin your analysis."
        )

    st.divider()

    st.subheader("Available analytical domains")

    domains = [
        "Data validation and cleaning",
        "Descriptive statistics",
        "Statistical hypothesis testing",
        "Regression and prediction",
        "Correlation analysis",
        "Sampling and sampling diagnostics",
        "Outlier detection",
        "Principal component analysis",
        "Clustering",
        "Time series analysis",
        "Bootstrap resampling",
        "Effect size calculation",
        "Statistical power",
        "Data profiling",
        "Reporting and export",
    ]

    for domain in domains:
        st.markdown(f"- {domain}")


# ============================================================
# DATA
# ============================================================

elif navigation == "📂 Data":

    st.header("Dataset management")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    if uploaded_file is not None:

        try:

            df = load_uploaded_file(uploaded_file)

            st.session_state.data = df.copy()
            st.session_state.original_data = df.copy()

            st.success("Dataset successfully loaded.")

        except Exception as e:
            show_error(e)

    df = get_data()

    if df is not None:

        st.subheader("Dataset preview")
        display_dataframe(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Observations", df.shape[0])
        col2.metric("Variables", df.shape[1])
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum()/1024:.2f} KB")

        st.subheader("Column information")

        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Missing": df.isna().sum().values,
                "Unique": df.nunique().values,
            }
        )

        display_dataframe(info_df)

        st.subheader("Dataset preview selection")

        rows = st.slider(
            "Number of rows to display",
            min_value=1,
            max_value=max(1, min(100, len(df))),
            value=min(10, len(df)),
        )

        display_dataframe(df.head(rows))

        st.subheader("Column selection")

        selected_columns = st.multiselect(
            "Columns to display",
            df.columns.tolist(),
            default=df.columns.tolist(),
        )

        if selected_columns:
            display_dataframe(df[selected_columns])

        st.subheader("Dataset reset")

        if st.button("Restore original dataset"):

            if st.session_state.original_data is not None:

                st.session_state.data = (
                    st.session_state.original_data.copy()
                )

                st.success("Dataset restored.")


# ============================================================
# CLEANING
# ============================================================

elif navigation == "🧹 Data cleaning":

    st.header("Data cleaning")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        st.subheader("Missing values")

        missing = df.isna().sum()

        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing values": missing.values,
                "Missing percentage": (
                    missing.values / max(len(df), 1) * 100
                ),
            }
        )

        display_dataframe(missing_df)

        action = st.selectbox(
            "Missing value treatment",
            [
                "Remove rows containing missing values",
                "Remove columns containing only missing values",
                "Fill numeric missing values with mean",
                "Fill numeric missing values with median",
                "Fill missing values with zero",
                "Fill missing values with a custom value",
            ],
        )

        custom_value = None

        if action == "Fill missing values with a custom value":
            custom_value = st.text_input("Replacement value")

        if st.button("Apply missing value treatment"):

            new_df = df.copy()

            try:

                if action == "Remove rows containing missing values":
                    new_df = new_df.dropna()

                elif action == "Remove columns containing only missing values":
                    new_df = new_df.dropna(axis=1, how="all")

                elif action == "Fill numeric missing values with mean":

                    for col in numeric_columns(new_df):
                        new_df[col] = new_df[col].fillna(
                            new_df[col].mean()
                        )

                elif action == "Fill numeric missing values with median":

                    for col in numeric_columns(new_df):
                        new_df[col] = new_df[col].fillna(
                            new_df[col].median()
                        )

                elif action == "Fill missing values with zero":
                    new_df = new_df.fillna(0)

                elif action == "Fill missing values with a custom value":
                    new_df = new_df.fillna(custom_value)

                st.session_state.data = new_df

                record_result("Missing value treatment", action)

                st.success("Cleaning applied.")

            except Exception as e:
                show_error(e)

        st.divider()

        st.subheader("Duplicates")

        duplicates = df.duplicated().sum()

        st.metric("Duplicate rows", int(duplicates))

        if st.button("Remove duplicate rows"):

            new_df = df.drop_duplicates()

            st.session_state.data = new_df

            st.success(
                f"{duplicates} duplicate rows removed."
            )

        st.divider()

        st.subheader("Column removal")

        columns_to_remove = st.multiselect(
            "Select columns to remove",
            df.columns.tolist(),
        )

        if st.button("Remove selected columns"):

            new_df = df.drop(columns=columns_to_remove)

            st.session_state.data = new_df

            st.success("Columns removed.")

        st.divider()

        st.subheader("Numeric data conversion")

        columns_to_convert = st.multiselect(
            "Select columns to convert to numeric",
            df.columns.tolist(),
        )

        if st.button("Convert selected columns"):

            new_df = df.copy()

            for col in columns_to_convert:
                new_df[col] = pd.to_numeric(
                    new_df[col],
                    errors="coerce",
                )

            st.session_state.data = new_df

            st.success("Conversion completed.")

        st.divider()

        st.subheader("Cleaning preview")

        display_dataframe(st.session_state.data)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

elif navigation == "📊 Descriptive statistics":

    st.header("Descriptive statistics")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if not nums:

            st.warning("No numeric columns detected.")

        else:

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            if selected:

                result = df[selected].describe().T

                result["variance"] = df[selected].var()
                result["skewness"] = df[selected].skew()
                result["kurtosis"] = df[selected].kurtosis()

                display_dataframe(result)

                download_csv(result.reset_index())

                st.subheader("Distribution")

                variable = st.selectbox(
                    "Variable",
                    selected,
                )

                st.histogram(df[variable].dropna())

                st.subheader("Box plot")

                st.boxplot = None

                st.write(
                    df[variable].describe()
                )

                st.subheader("Quantiles")

                quantiles = df[selected].quantile(
                    [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
                )

                display_dataframe(quantiles)


# ============================================================
# STATISTICAL TESTS
# ============================================================

elif navigation == "🧪 Statistical tests":

    st.header("Statistical hypothesis testing")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if not nums:

            st.warning("Numeric columns required.")

        else:

            test_type = st.selectbox(
                "Test",
                [
                    "One-sample t-test",
                    "Independent t-test",
                    "Paired t-test",
                    "Mann-Whitney U",
                    "Wilcoxon",
                    "One-way ANOVA",
                    "Kruskal-Wallis",
                    "Shapiro-Wilk normality",
                ],
            )

            alpha = st.slider(
                "Significance level",
                0.01,
                0.10,
                0.05,
            )

            if test_type == "One-sample t-test":

                col = st.selectbox("Variable", nums)

                mu = st.number_input(
                    "Reference mean",
                    value=0.0,
                )

                if st.button("Run test"):

                    x = clean_numeric(df[col])

                    result = stats.ttest_1samp(
                        x,
                        mu,
                    )

                    st.metric("Statistic", result.statistic)
                    st.metric("p-value", result.pvalue)

                    st.write(
                        "Reject H0"
                        if result.pvalue < alpha
                        else "Do not reject H0"
                    )

            elif test_type == "Independent t-test":

                col = st.selectbox("Numeric variable", nums)

                cats = categorical_columns(df)

                if cats:

                    group = st.selectbox(
                        "Grouping variable",
                        cats,
                    )

                    groups = df[group].dropna().unique()

                    if len(groups) >= 2:

                        g1 = st.selectbox(
                            "Group 1",
                            groups,
                        )

                        g2 = st.selectbox(
                            "Group 2",
                            [
                                x for x in groups
                                if x != g1
                            ],
                        )

                        if st.button("Run test"):

                            x = clean_numeric(
                                df.loc[df[group] == g1, col]
                            )

                            y = clean_numeric(
                                df.loc[df[group] == g2, col]
                            )

                            result = stats.ttest_ind(
                                x,
                                y,
                                equal_var=False,
                            )

                            st.metric(
                                "Statistic",
                                result.statistic,
                            )

                            st.metric(
                                "p-value",
                                result.pvalue,
                            )

                else:
                    st.warning("Categorical variable required.")

            elif test_type == "Paired t-test":

                col1 = st.selectbox(
                    "First variable",
                    nums,
                )

                col2 = st.selectbox(
                    "Second variable",
                    [
                        c for c in nums
                        if c != col1
                    ],
                )

                if st.button("Run test"):

                    paired = df[[col1, col2]].dropna()

                    result = stats.ttest_rel(
                        paired[col1],
                        paired[col2],
                    )

                    st.metric(
                        "Statistic",
                        result.statistic,
                    )

                    st.metric(
                        "p-value",
                        result.pvalue,
                    )

            elif test_type == "Mann-Whitney U":

                col = st.selectbox("Numeric variable", nums)

                cats = categorical_columns(df)

                if cats:

                    group = st.selectbox(
                        "Grouping variable",
                        cats,
                    )

                    groups = df[group].dropna().unique()

                    if len(groups) >= 2:

                        g1 = st.selectbox(
                            "Group 1",
                            groups,
                        )

                        g2 = st.selectbox(
                            "Group 2",
                            [x for x in groups if x != g1],
                        )

                        if st.button("Run test"):

                            x = clean_numeric(
                                df.loc[df[group] == g1, col]
                            )

                            y = clean_numeric(
                                df.loc[df[group] == g2, col]
                            )

                            result = stats.mannwhitneyu(
                                x,
                                y,
                                alternative="two-sided",
                            )

                            st.metric(
                                "Statistic",
                                result.statistic,
                            )

                            st.metric(
                                "p-value",
                                result.pvalue,
                            )

            elif test_type == "Wilcoxon":

                col1 = st.selectbox(
                    "First variable",
                    nums,
                )

                col2 = st.selectbox(
                    "Second variable",
                    [x for x in nums if x != col1],
                )

                if st.button("Run test"):

                    paired = df[[col1, col2]].dropna()

                    result = stats.wilcoxon(
                        paired[col1],
                        paired[col2],
                    )

                    st.metric(
                        "Statistic",
                        result.statistic,
                    )

                    st.metric(
                        "p-value",
                        result.pvalue,
                    )

            elif test_type == "One-way ANOVA":

                col = st.selectbox(
                    "Numeric variable",
                    nums,
                )

                cats = categorical_columns(df)

                if cats:

                    group = st.selectbox(
                        "Grouping variable",
                        cats,
                    )

                    data = [
                        clean_numeric(
                            df.loc[df[group] == g, col]
                        )
                        for g in df[group].dropna().unique()
                    ]

                    data = [
                        x for x in data
                        if len(x) > 0
                    ]

                    if st.button("Run test"):

                        if len(data) >= 2:

                            result = stats.f_oneway(*data)

                            st.metric(
                                "Statistic",
                                result.statistic,
                            )

                            st.metric(
                                "p-value",
                                result.pvalue,
                            )

                        else:
                            st.error("At least two groups required.")

            elif test_type == "Kruskal-Wallis":

                col = st.selectbox(
                    "Numeric variable",
                    nums,
                )

                cats = categorical_columns(df)

                if cats:

                    group = st.selectbox(
                        "Grouping variable",
                        cats,
                    )

                    data = [
                        clean_numeric(
                            df.loc[df[group] == g, col]
                        )
                        for g in df[group].dropna().unique()
                    ]

                    data = [
                        x for x in data
                        if len(x) > 0
                    ]

                    if st.button("Run test"):

                        if len(data) >= 2:

                            result = stats.kruskal(*data)

                            st.metric(
                                "Statistic",
                                result.statistic,
                            )

                            st.metric(
                                "p-value",
                                result.pvalue,
                            )

            elif test_type == "Shapiro-Wilk normality":

                col = st.selectbox(
                    "Variable",
                    nums,
                )

                if st.button("Run test"):

                    x = clean_numeric(df[col])

                    if 3 <= len(x) <= 5000:

                        result = stats.shapiro(x)

                        st.metric(
                            "Statistic",
                            result.statistic,
                        )

                        st.metric(
                            "p-value",
                            result.pvalue,
                        )

                    else:

                        st.warning(
                            "Shapiro-Wilk requires "
                            "3 to 5000 observations."
                        )


# ============================================================
# REGRESSION
# ============================================================

elif navigation == "📈 Regression":

    st.header("Regression analysis")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            target = st.selectbox(
                "Target variable",
                nums,
            )

            features = st.multiselect(
                "Predictor variables",
                [c for c in nums if c != target],
            )

            if features:

                model_type = st.radio(
                    "Model",
                    [
                        "Linear regression",
                    ],
                )

                test_size = st.slider(
                    "Test size",
                    0.1,
                    0.5,
                    0.2,
                )

                if st.button("Fit model"):

                    data = df[
                        features + [target]
                    ].dropna()

                    X = data[features]
                    y = data[target]

                    X_train, X_test, y_train, y_test = (
                        train_test_split(
                            X,
                            y,
                            test_size=test_size,
                            random_state=42,
                        )
                    )

                    model = LinearRegression()

                    model.fit(
                        X_train,
                        y_train,
                    )

                    predictions = model.predict(X_test)

                    st.metric(
                        "R²",
                        round(
                            r2_score(
                                y_test,
                                predictions,
                            ),
                            4,
                        ),
                    )

                    st.metric(
                        "RMSE",
                        round(
                            np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    predictions,
                                )
                            ),
                            4,
                        ),
                    )

                    coefficient_df = pd.DataFrame(
                        {
                            "Variable": features,
                            "Coefficient": model.coef_,
                        }
                    )

                    display_dataframe(coefficient_df)

                    result_df = pd.DataFrame(
                        {
                            "Observed": y_test.values,
                            "Predicted": predictions,
                        }
                    )

                    display_dataframe(result_df)

                    download_csv(
                        result_df,
                        "regression_predictions.csv",
                    )

                    record_result(
                        "Regression",
                        {
                            "R2": r2_score(
                                y_test,
                                predictions,
                            ),
                            "RMSE": np.sqrt(
                                mean_squared_error(
                                    y_test,
                                    predictions,
                                )
                            ),
                        },
                    )

        else:

            st.warning(
                "At least two numeric columns required."
            )


# ============================================================
# CORRELATION
# ============================================================

elif navigation == "🔗 Correlation":

    st.header("Correlation analysis")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            method = st.selectbox(
                "Method",
                [
                    "pearson",
                    "spearman",
                    "kendall",
                ],
            )

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            if len(selected) >= 2:

                corr = df[selected].corr(
                    method=method,
                )

                st.subheader("Correlation matrix")

                display_dataframe(corr)

                download_csv(
                    corr.reset_index(),
                    "correlation_matrix.csv",
                )

                st.subheader("Pairwise correlations")

                pairs = []

                for i, col1 in enumerate(selected):

                    for col2 in selected[i + 1:]:

                        x = df[[col1, col2]].dropna()

                        if len(x) >= 2:

                            if method == "pearson":
                                result = stats.pearsonr(
                                    x[col1],
                                    x[col2],
                                )

                            elif method == "spearman":
                                result = stats.spearmanr(
                                    x[col1],
                                    x[col2],
                                )

                            else:
                                result = stats.kendalltau(
                                    x[col1],
                                    x[col2],
                                )

                            pairs.append(
                                {
                                    "Variable 1": col1,
                                    "Variable 2": col2,
                                    "Correlation": result.statistic,
                                    "p-value": result.pvalue,
                                }
                            )

                pair_df = pd.DataFrame(pairs)

                display_dataframe(pair_df)

        else:

            st.warning(
                "At least two numeric columns required."
            )


# ============================================================
# SAMPLING
# ============================================================

elif navigation == "🎯 Sampling":

    st.header("Sampling")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        method = st.selectbox(
            "Sampling method",
            [
                "Random sampling",
                "Stratified sampling",
            ],
        )

        sample_size = st.number_input(
            "Sample size",
            min_value=1,
            max_value=len(df),
            value=min(
                max(1, int(len(df) * 0.2)),
                len(df),
            ),
        )

        if method == "Stratified sampling":

            cats = categorical_columns(df)

            if cats:

                stratify_column = st.selectbox(
                    "Stratification column",
                    cats,
                )

            else:

                stratify_column = None

                st.warning(
                    "No categorical column detected."
                )

        if st.button("Generate sample"):

            try:

                if method == "Random sampling":

                    sample = df.sample(
                        n=int(sample_size),
                        random_state=42,
                    )

                else:

                    if stratify_column is None:
                        raise ValueError(
                            "Stratification variable required."
                        )

                    data = df.dropna(
                        subset=[stratify_column]
                    )

                    proportions = (
                        data[stratify_column]
                        .value_counts(normalize=True)
                    )

                    sample_parts = []

                    for category, proportion in proportions.items():

                        group = data[
                            data[stratify_column] == category
                        ]

                        n = max(
                            1,
                            round(
                                int(sample_size)
                                * proportion
                            ),
                        )

                        n = min(
                            n,
                            len(group),
                        )

                        sample_parts.append(
                            group.sample(
                                n=n,
                                random_state=42,
                            )
                        )

                    sample = pd.concat(
                        sample_parts,
                    )

                    if len(sample) > int(sample_size):

                        sample = sample.sample(
                            n=int(sample_size),
                            random_state=42,
                        )

                st.subheader("Sample")

                display_dataframe(sample)

                st.metric(
                    "Sample size",
                    len(sample),
                )

                download_csv(
                    sample,
                    "biodatalab_sample.csv",
                )

                st.subheader("Sample composition")

                if method == "Stratified sampling":

                    st.write(
                        "Population proportions"
                    )

                    display_dataframe(
                        df[stratify_column]
                        .value_counts(normalize=True)
                    )

                    st.write(
                        "Sample proportions"
                    )

                    display_dataframe(
                        sample[stratify_column]
                        .value_counts(normalize=True)
                    )

                record_result(
                    "Sampling",
                    {
                        "method": method,
                        "sample_size": len(sample),
                    },
                )

            except Exception as e:
                show_error(e)


# ============================================================
# OUTLIERS
# ============================================================

elif navigation == "📉 Outliers":

    st.header("Outlier detection")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox(
                "Variable",
                nums,
            )

            method = st.selectbox(
                "Detection method",
                [
                    "IQR",
                    "Z-score",
                ],
            )

            x = clean_numeric(df[col])

            if method == "IQR":

                q1 = x.quantile(0.25)
                q3 = x.quantile(0.75)

                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                mask = (
                    (df[col] < lower)
                    | (df[col] > upper)
                )

            else:

                z = (
                    df[col] - df[col].mean()
                ) / df[col].std()

                mask = z.abs() > 3

            outliers = df[mask]

            st.metric(
                "Detected outliers",
                len(outliers),
            )

            display_dataframe(outliers)

            if st.button("Remove detected outliers"):

                st.session_state.data = df[
                    ~mask
                ].copy()

                st.success(
                    "Outliers removed from the dataset."
                )

            download_csv(
                outliers,
                "outliers.csv",
            )


# ============================================================
# PCA
# ============================================================

elif navigation == "🧬 PCA":

    st.header("Principal component analysis")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            n_components = st.slider(
                "Components",
                2,
                min(10, len(selected)),
                2,
            ) if len(selected) >= 2 else 2

            if len(selected) >= 2:

                if st.button("Run PCA"):

                    data = df[selected].dropna()

                    scaled = StandardScaler().fit_transform(
                        data
                    )

                    pca = PCA(
                        n_components=n_components,
                    )

                    transformed = pca.fit_transform(
                        scaled
                    )

                    result_df = pd.DataFrame(
                        transformed,
                        columns=[
                            f"PC{i+1}"
                            for i in range(n_components)
                        ],
                    )

                    st.subheader("Transformed dataset")

                    display_dataframe(result_df)

                    st.subheader("Explained variance")

                    variance_df = pd.DataFrame(
                        {
                            "Component": [
                                f"PC{i+1}"
                                for i in range(n_components)
                            ],
                            "Variance explained": (
                                pca.explained_variance_ratio_
                            ),
                            "Cumulative variance": (
                                np.cumsum(
                                    pca.explained_variance_ratio_
                                )
                            ),
                        }
                    )

                    display_dataframe(variance_df)

                    st.subheader("Loadings")

                    loadings = pd.DataFrame(
                        pca.components_.T,
                        index=selected,
                        columns=[
                            f"PC{i+1}"
                            for i in range(n_components)
                        ],
                    )

                    display_dataframe(loadings)

                    download_csv(
                        result_df,
                        "pca_results.csv",
                    )

                    download_csv(
                        loadings.reset_index(),
                        "pca_loadings.csv",
                    )

        else:

            st.warning(
                "At least two numeric variables required."
            )


# ============================================================
# CLUSTERING
# ============================================================

elif navigation == "🔬 Clustering":

    st.header("K-means clustering")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            k = st.slider(
                "Number of clusters",
                2,
                10,
                3,
            )

            if len(selected) >= 2:

                if st.button("Run clustering"):

                    data = df[selected].dropna()

                    scaled = StandardScaler().fit_transform(
                        data
                    )

                    model = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=10,
                    )

                    labels = model.fit_predict(
                        scaled
                    )

                    result_df = data.copy()

                    result_df["Cluster"] = labels

                    st.subheader("Cluster assignments")

                    display_dataframe(result_df)

                    st.subheader("Cluster sizes")

                    display_dataframe(
                        result_df["Cluster"]
                        .value_counts()
                        .sort_index()
                    )

                    st.subheader("Cluster centers")

                    centers = pd.DataFrame(
                        model.cluster_centers_,
                        columns=selected,
                    )

                    display_dataframe(centers)

                    st.metric(
                        "Inertia",
                        model.inertia_,
                    )

                    download_csv(
                        result_df,
                        "clustering_results.csv",
                    )

        else:

            st.warning(
                "At least two numeric variables required."
            )


# ============================================================
# TIME SERIES
# ============================================================

elif navigation == "📅 Time series":

    st.header("Time series analysis")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        date_columns = [
            col for col in df.columns
            if (
                pd.api.types.is_datetime64_any_dtype(
                    df[col]
                )
            )
        ]

        date_columns += [
            col for col in df.columns
            if (
                col not in date_columns
                and (
                    "date" in col.lower()
                    or "time" in col.lower()
                )
            ]

        )

        date_columns = list(
            dict.fromkeys(date_columns)
        )

        nums = numeric_columns(df)

        if date_columns and nums:

            date_col = st.selectbox(
                "Date column",
                date_columns,
            )

            value_col = st.selectbox(
                "Value column",
                nums,
            )

            if st.button("Analyze time series"):

                data = df[
                    [date_col, value_col]
                ].copy()

                data[date_col] = pd.to_datetime(
                    data[date_col],
                    errors="coerce",
                )

                data[value_col] = pd.to_numeric(
                    data[value_col],
                    errors="coerce",
                )

                data = data.dropna()

                data = data.sort_values(
                    date_col
                )

                data = data.set_index(
                    date_col
                )

                st.subheader("Time series")

                st.line_chart(
                    data[value_col]
                )

                st.subheader("Rolling mean")

                window = st.slider(
                    "Rolling window",
                    2,
                    max(
                        2,
                        min(
                            100,
                            len(data),
                        ),
                    ),
                    min(
                        7,
                        max(
                            2,
                            len(data),
                        ),
                    ),
                )

                rolling = data[
                    value_col
                ].rolling(
                    window=window,
                ).mean()

                st.line_chart(
                    rolling
                )

                st.subheader("Rolling standard deviation")

                rolling_std = data[
                    value_col
                ].rolling(
                    window=window,
                ).std()

                st.line_chart(
                    rolling_std
                )

                result = pd.DataFrame(
                    {
                        "Original": data[value_col],
                        "Rolling mean": rolling,
                        "Rolling standard deviation": rolling_std,
                    }
                )

                download_csv(
                    result.reset_index(),
                    "time_series_results.csv",
                )

        else:

            st.warning(
                "A date column and numeric variable are required."
            )


# ============================================================
# BOOTSTRAP
# ============================================================

elif navigation == "⚡ Bootstrap":

    st.header("Bootstrap resampling")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox(
                "Variable",
                nums,
            )

            iterations = st.number_input(
                "Iterations",
                100,
                10000,
                1000,
                step=100,
            )

            confidence = st.selectbox(
                "Confidence level",
                [90, 95, 99],
                index=1,
            )

            statistic = st.selectbox(
                "Statistic",
                [
                    "Mean",
                    "Median",
                ],
            )

            if st.button("Run bootstrap"):

                x = clean_numeric(df[col]).values

                if len(x) > 0:

                    results = []

                    for _ in range(
                        int(iterations)
                    ):

                        sample = np.random.choice(
                            x,
                            size=len(x),
                            replace=True,
                        )

                        if statistic == "Mean":
                            results.append(
                                np.mean(sample)
                            )

                        else:
                            results.append(
                                np.median(sample)
                            )

                    results = np.asarray(
                        results
                    )

                    alpha = (
                        1 - confidence / 100
                    )

                    lower = np.quantile(
                        results,
                        alpha / 2,
                    )

                    upper = np.quantile(
                        results,
                        1 - alpha / 2,
                    )

                    st.metric(
                        f"Bootstrap {statistic}",
                        np.mean(results),
                    )

                    st.metric(
                        "Lower confidence bound",
                        lower,
                    )

                    st.metric(
                        "Upper confidence bound",
                        upper,
                    )

                    result_df = pd.DataFrame(
                        {
                            "Bootstrap statistic": results
                        }
                    )

                    display_dataframe(
                        result_df.head(100)
                    )

                    download_csv(
                        result_df,
                        "bootstrap_results.csv",
                    )


# ============================================================
# EFFECT SIZE
# ============================================================

elif navigation == "📐 Effect size":

    st.header("Effect size")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        cats = categorical_columns(df)

        if nums and cats:

            col = st.selectbox(
                "Numeric variable",
                nums,
            )

            group = st.selectbox(
                "Grouping variable",
                cats,
            )

            groups = df[group].dropna().unique()

            if len(groups) >= 2:

                g1 = st.selectbox(
                    "Group 1",
                    groups,
                )

                g2 = st.selectbox(
                    "Group 2",
                    [g for g in groups if g != g1],
                )

                if st.button("Calculate effect size"):

                    x = clean_numeric(
                        df.loc[df[group] == g1, col]
                    )

                    y = clean_numeric(
                        df.loc[df[group] == g2, col]
                    )

                    if len(x) > 1 and len(y) > 1:

                        pooled_sd = np.sqrt(
                            (
                                (len(x) - 1) * x.var()
                                + (len(y) - 1) * y.var()
                            )
                            / (
                                len(x) + len(y) - 2
                            )
                        )

                        if pooled_sd != 0:

                            cohens_d = (
                                x.mean() - y.mean()
                            ) / pooled_sd

                            st.metric(
                                "Cohen's d",
                                cohens_d,
                            )

                            st.write(
                                "Interpretation is context-dependent. "
                                "Common reference values of 0.2, 0.5 "
                                "and 0.8 are only rough guidelines."
                            )

                        else:

                            st.warning(
                                "Pooled standard deviation is zero."
                            )

        else:

            st.warning(
                "Numeric and categorical variables required."
            )


# ============================================================
# POWER
# ============================================================

elif navigation == "🔋 Statistical power":

    st.header("Statistical power estimation")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox(
                "Variable",
                nums,
            )

            mu = st.number_input(
                "Reference mean",
                value=0.0,
            )

            alpha = st.slider(
                "Alpha",
                0.01,
                0.10,
                0.05,
            )

            x = clean_numeric(df[col])

            if len(x) > 1:

                effect = (
                    x.mean() - mu
                ) / x.std()

                st.metric(
                    "Observed standardized effect",
                    effect,
                )

                st.write(
                    "This is a descriptive estimate. "
                    "Post-hoc power based on observed effects "
                    "should not be used as a substitute for "
                    "confidence intervals or prospective design."
                )

                st.metric(
                    "Sample size",
                    len(x),
                )

                st.metric(
                    "Mean",
                    x.mean(),
                )

                st.metric(
                    "Standard deviation",
                    x.std(),
                )


# ============================================================
# PROFILING
# ============================================================

elif navigation == "📋 Data profiling":

    st.header("Data profiling")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        profile = []

        for col in df.columns:

            s = df[col]

            profile.append(
                {
                    "Column": col,
                    "Type": str(s.dtype),
                    "Rows": len(s),
                    "Missing": int(s.isna().sum()),
                    "Missing (%)": (
                        s.isna().sum()
                        / max(len(s), 1)
                        * 100
                    ),
                    "Unique": s.nunique(),
                    "Memory (bytes)": (
                        s.memory_usage(deep=True)
                    ),
                }
            )

        profile_df = pd.DataFrame(
            profile
        )

        display_dataframe(
            profile_df
        )

        st.subheader("Dataset summary")

        st.write(
            {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Duplicate rows": int(
                    df.duplicated().sum()
                ),
                "Missing cells": int(
                    df.isna().sum().sum()
                ),
                "Numeric columns": len(
                    numeric_columns(df)
                ),
                "Categorical columns": len(
                    categorical_columns(df)
                ),
            }
        )

        download_csv(
            profile_df,
            "data_profile.csv",
        )


# ============================================================
# REPORTS
# ============================================================

elif navigation == "📑 Reports":

    st.header("Analysis report")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        report = {
            "project": "BioDataLab",
            "generated_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "missing_values": {
                col: int(value)
                for col, value in df.isna().sum().items()
            },
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
            "descriptive_statistics": (
                df.describe()
                .to_dict()
            ),
            "analysis_history": (
                st.session_state.history
            ),
        }

        st.subheader("Report summary")

        st.write(
            {
                "Rows": report["rows"],
                "Columns": report["columns"],
                "Missing values": int(
                    df.isna().sum().sum()
                ),
                "Duplicates": report["duplicate_rows"],
            }
        )

        st.subheader("Descriptive statistics")

        display_dataframe(
            df.describe()
        )

        st.subheader("Analysis history")

        if st.session_state.history:

            display_dataframe(
                pd.DataFrame(
                    st.session_state.history
                )
            )

        else:

            st.info(
                "No analysis recorded during this session."
            )

        download_json(
            report,
            "biodatalab_report.json",
        )


# ============================================================
# EXPORT
# ============================================================

elif navigation == "📤 Export":

    st.header("Export")

    df = get_data()

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        st.subheader("Current dataset")

        display_dataframe(df)

        download_csv(
            df,
            "biodatalab_dataset.csv",
        )

        st.subheader("Excel export")

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                sheet_name="Dataset",
                index=False,
            )

            if numeric_columns(df):

                df.describe().to_excel(
                    writer,
                    sheet_name="Statistics",
                )

        st.download_button(
            label="📥 Download Excel workbook",
            data=excel_buffer.getvalue(),
            file_name="biodatalab.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

        st.subheader("Analysis results")

        if st.session_state.results:

            download_json(
                st.session_state.results,
                "biodatalab_analysis_results.json",
            )

        else:

            st.info(
                "No analysis results recorded."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BioDataLab | Open-source scientific data analysis | "
    "Designed for reproducibility and accessibility"
)
