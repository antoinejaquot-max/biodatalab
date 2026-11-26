"""BioDataLab Streamlit application."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BioDataLab",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 BioDataLab")
st.caption(
    "Open-source platform for biological, agricultural "
    "and scientific data analysis."
)

# ============================================================
# HELPERS
# ============================================================


def record(name, results):
    st.session_state.setdefault("analysis_results", [])
    st.session_state["analysis_results"].append(
        {
            "analysis": name,
            "results": results,
        }
    )


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def clean_numeric(df):
    return df.select_dtypes(include=np.number).dropna()


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return None
    except (TypeError, ValueError):
        return None


def display_result_dict(results):
    for key, value in results.items():
        if isinstance(value, float):
            st.write(f"**{key}:** {value:.6g}")
        else:
            st.write(f"**{key}:** {value}")


def export_results():
    results = st.session_state.get("analysis_results", [])
    return json.dumps(results, indent=2, default=str)


# ============================================================
# DATA LOADING
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Import a dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state["dataset"] = df

    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")

if "dataset" not in st.session_state:

    st.info("Upload a CSV or Excel dataset to begin.")

    st.markdown(
        """
        ### Available features

        - Data exploration
        - Data cleaning
        - Descriptive statistics
        - Statistical tests
        - Regression
        - Correlation
        - Sampling
        - Outlier detection
        - PCA
        - Clustering
        - Time series
        - Bootstrap
        - Effect sizes
        - Power analysis
        - Profiling
        - Reporting
        - Export
        """
    )

    st.stop()

df = st.session_state["dataset"]

# ============================================================
# SIDEBAR
# ============================================================

pages = [
    "Overview",
    "Dataset explorer",
    "Data cleaning",
    "Descriptive statistics",
    "Statistical tests",
    "Regression",
    "Correlation",
    "Sampling",
    "Outlier detection",
    "PCA",
    "Clustering",
    "Time series",
    "Bootstrap",
    "Effect size",
    "Power analysis",
    "Profiling",
    "Reports",
    "Export",
]

page = st.sidebar.radio("Navigation", pages)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Dataset overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isna().sum().sum()))
    col4.metric("Numeric variables", len(numeric_columns(df)))

    st.subheader("Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Data types")

    st.dataframe(
        pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Missing": df.isna().sum().values,
                "Unique": df.nunique().values,
            }
        ),
        use_container_width=True,
    )

# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset explorer":

    st.header("Dataset explorer")

    st.dataframe(df, use_container_width=True)

    st.subheader("Column information")

    selected_column = st.selectbox(
        "Select a column",
        df.columns,
    )

    st.write("**Data type:**", df[selected_column].dtype)
    st.write("**Missing values:**", int(df[selected_column].isna().sum()))
    st.write("**Unique values:**", int(df[selected_column].nunique()))

    st.write("**Value counts**")

    st.dataframe(
        df[selected_column].value_counts(dropna=False),
        use_container_width=True,
    )

# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data cleaning":

    st.header("Data cleaning")

    st.write("Current dataset")

    st.dataframe(df.head(), use_container_width=True)

    cleaned = df.copy()

    st.subheader("Missing values")

    missing_action = st.selectbox(
        "Missing values treatment",
        [
            "Keep unchanged",
            "Remove rows",
            "Fill with mean",
            "Fill with median",
            "Fill with zero",
        ],
    )

    if missing_action == "Remove rows":
        cleaned = cleaned.dropna()

    elif missing_action == "Fill with mean":

        for col in numeric_columns(cleaned):
            cleaned[col] = cleaned[col].fillna(
                cleaned[col].mean()
            )

    elif missing_action == "Fill with median":

        for col in numeric_columns(cleaned):
            cleaned[col] = cleaned[col].fillna(
                cleaned[col].median()
            )

    elif missing_action == "Fill with zero":
        cleaned = cleaned.fillna(0)

    st.subheader("Duplicates")

    remove_duplicates = st.checkbox(
        "Remove duplicate rows"
    )

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    st.subheader("Cleaning preview")

    st.dataframe(cleaned, use_container_width=True)

    if st.button("Apply cleaning"):

        st.session_state["dataset"] = cleaned

        record(
            "Cleaning",
            {
                "rows_before": len(df),
                "rows_after": len(cleaned),
                "missing_before": int(df.isna().sum().sum()),
                "missing_after": int(
                    cleaned.isna().sum().sum()
                ),
            },
        )

        st.success("Cleaning applied.")

        st.rerun()

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

elif page == "Descriptive statistics":

    st.header("Descriptive statistics")

    cols = numeric_columns(df)

    if not cols:
        st.warning("No numeric variables detected.")
        st.stop()

    selected = st.multiselect(
        "Variables",
        cols,
        default=cols,
    )

    if selected:

        descriptive = df[selected].describe().T

        descriptive["variance"] = df[selected].var()
        descriptive["skewness"] = df[selected].skew()
        descriptive["kurtosis"] = df[selected].kurtosis()

        st.dataframe(
            descriptive,
            use_container_width=True,
        )

        record(
            "Descriptive statistics",
            descriptive.to_dict(),
        )

# ============================================================
# STATISTICAL TESTS
# ============================================================

elif page == "Statistical tests":

    st.header("Statistical tests")

    cols = numeric_columns(df)

    if len(cols) < 1:
        st.warning("Numeric variables required.")
        st.stop()

    test_type = st.selectbox(
        "Test",
        [
            "One-sample t-test",
            "Independent t-test",
            "Paired t-test",
            "Mann-Whitney",
            "Wilcoxon",
            "Kruskal-Wallis",
            "One-way ANOVA",
            "Pearson correlation test",
            "Spearman correlation test",
            "Normality test",
        ],
    )

    col1 = st.selectbox("Variable 1", cols)

    col2 = None

    if test_type != "One-sample t-test":
        col2 = st.selectbox(
            "Variable 2",
            [c for c in cols if c != col1],
        )

    results = {}

    if test_type == "One-sample t-test":

        value = st.number_input(
            "Reference value",
            value=0.0,
        )

        data = df[col1].dropna()

        if len(data) >= 2:

            result = stats.ttest_1samp(
                data,
                value,
            )

            results = {
                "test": test_type,
                "t_statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Independent t-test":

        a = df[col1].dropna()
        b = df[col2].dropna()

        if len(a) >= 2 and len(b) >= 2:

            result = stats.ttest_ind(
                a,
                b,
                equal_var=False,
            )

            results = {
                "test": test_type,
                "t_statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Paired t-test":

        paired = df[[col1, col2]].dropna()

        if len(paired) >= 2:

            result = stats.ttest_rel(
                paired[col1],
                paired[col2],
            )

            results = {
                "test": test_type,
                "t_statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Mann-Whitney":

        a = df[col1].dropna()
        b = df[col2].dropna()

        if len(a) and len(b):

            result = stats.mannwhitneyu(
                a,
                b,
                alternative="two-sided",
            )

            results = {
                "test": test_type,
                "statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Wilcoxon":

        paired = df[[col1, col2]].dropna()

        if len(paired) >= 2:

            result = stats.wilcoxon(
                paired[col1],
                paired[col2],
            )

            results = {
                "test": test_type,
                "statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Kruskal-Wallis":

        groups = [
            df[col].dropna()
            for col in cols
        ]

        groups = [
            g for g in groups
            if len(g) > 0
        ]

        if len(groups) >= 2:

            result = stats.kruskal(*groups)

            results = {
                "test": test_type,
                "statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "One-way ANOVA":

        groups = [
            df[col].dropna()
            for col in cols
        ]

        groups = [
            g for g in groups
            if len(g) > 0
        ]

        if len(groups) >= 2:

            result = stats.f_oneway(*groups)

            results = {
                "test": test_type,
                "F_statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Pearson correlation test":

        paired = df[[col1, col2]].dropna()

        if len(paired) >= 3:

            result = stats.pearsonr(
                paired[col1],
                paired[col2],
            )

            results = {
                "test": test_type,
                "correlation": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Spearman correlation test":

        paired = df[[col1, col2]].dropna()

        if len(paired) >= 3:

            result = stats.spearmanr(
                paired[col1],
                paired[col2],
            )

            results = {
                "test": test_type,
                "correlation": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    elif test_type == "Normality test":

        data = df[col1].dropna()

        if 3 <= len(data) <= 5000:

            result = stats.shapiro(data)

            results = {
                "test": test_type,
                "W_statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

        elif len(data) > 5000:

            result = stats.normaltest(data)

            results = {
                "test": "D'Agostino normality test",
                "statistic": safe_float(result.statistic),
                "p_value": safe_float(result.pvalue),
            }

    if results:

        st.subheader("Results")

        display_result_dict(results)

        record(
            "Statistical test",
            results,
        )

# ============================================================
# REGRESSION
# ============================================================

elif page == "Regression":

    st.header("Regression")

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.warning("At least two numeric variables are required.")
        st.stop()

    target = st.selectbox(
        "Target variable",
        cols,
    )

    predictors = st.multiselect(
        "Predictors",
        [c for c in cols if c != target],
    )

    if predictors:

        regression_df = df[
            predictors + [target]
        ].dropna()

        if len(regression_df) >= 3:

            X = regression_df[predictors]
            y = regression_df[target]

            model = LinearRegression()

            model.fit(X, y)

            pred = model.predict(X)

            r2 = r2_score(y, pred)
            rmse = np.sqrt(
                mean_squared_error(y, pred)
            )
            mae = mean_absolute_error(y, pred)

            results = {
                "R2": safe_float(r2),
                "RMSE": safe_float(rmse),
                "MAE": safe_float(mae),
                "Intercept": safe_float(model.intercept_),
            }

            st.subheader("Performance")

            display_result_dict(results)

            st.subheader("Coefficients")

            coefficients = pd.DataFrame(
                {
                    "Variable": predictors,
                    "Coefficient": model.coef_,
                }
            )

            st.dataframe(
                coefficients,
                use_container_width=True,
            )

            st.subheader("Predictions")

            prediction_df = pd.DataFrame(
                {
                    "Observed": y.values,
                    "Predicted": pred,
                }
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
            )

            record(
                "Regression",
                {
                    "R2": safe_float(r2_score(y, pred)),
                    "RMSE": safe_float(
                        np.sqrt(
                            mean_squared_error(y, pred)
                        )
                    ),
                    "MAE": safe_float(
                        mean_absolute_error(y, pred)
                    ),
                    "Coefficients": dict(
                        zip(predictors, model.coef_.tolist())
                    ),
                },
            )

# ============================================================
# CORRELATION
# ============================================================

elif page == "Correlation":

    st.header("Correlation analysis")

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.warning("At least two numeric variables are required.")
        st.stop()

    method = st.selectbox(
        "Method",
        ["pearson", "spearman", "kendall"],
    )

    matrix = df[cols].corr(
        method=method,
    )

    st.dataframe(
        matrix,
        use_container_width=True,
    )

    st.subheader("Correlation matrix")

    st.dataframe(
        matrix.style.format("{:.3f}"),
        use_container_width=True,
    )

    record(
        "Correlation",
        {
            "method": method,
            "matrix": matrix.to_dict(),
        },
    )

# ============================================================
# SAMPLING
# ============================================================

elif page == "Sampling":

    st.header("Sampling")

    sample_size = st.slider(
        "Sample size",
        min_value=1,
        max_value=len(df),
        value=max(1, len(df) // 2),
    )

    random_state = st.number_input(
        "Random seed",
        value=42,
        step=1,
    )

    sampled = df.sample(
        n=sample_size,
        random_state=int(random_state),
    )

    st.dataframe(
        sampled,
        use_container_width=True,
    )

    st.download_button(
        "Download sample CSV",
        sampled.to_csv(index=False),
        file_name="sample.csv",
        mime="text/csv",
    )

# ============================================================
# OUTLIER DETECTION
# ============================================================

elif page == "Outlier detection":

    st.header("Outlier detection")

    cols = numeric_columns(df)

    if not cols:
        st.warning("No numeric variables detected.")
        st.stop()

    selected = st.selectbox(
        "Variable",
        cols,
    )

    method = st.selectbox(
        "Method",
        ["IQR", "Z-score"],
    )

    data = df[selected].dropna()

    if method == "IQR":

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = data[
            (data < lower) | (data > upper)
        ]

    else:

        z = np.abs(
            stats.zscore(data, nan_policy="omit")
        )

        outliers = data[
            z > 3
        ]

        lower = None
        upper = None

    st.metric(
        "Detected outliers",
        len(outliers),
    )

    st.dataframe(
        outliers.to_frame("Outlier"),
        use_container_width=True,
    )

    if method == "IQR":

        st.write("Lower bound:", lower)
        st.write("Upper bound:", upper)

    record(
        "Outlier detection",
        {
            "variable": selected,
            "method": method,
            "count": int(len(outliers)),
        },
    )

# ============================================================
# PCA
# ============================================================

elif page == "PCA":

    st.header("Principal Component Analysis")

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.warning("At least two numeric variables are required.")
        st.stop()

    selected = st.multiselect(
        "Variables",
        cols,
        default=cols,
    )

    if len(selected) >= 2:

        data = df[selected].dropna()

        if len(data) >= 2:

            scaled = StandardScaler().fit_transform(data)

            pca = PCA()

            transformed = pca.fit_transform(scaled)

            st.subheader("Explained variance")

            variance = pd.DataFrame(
                {
                    "Component": [
                        f"PC{i + 1}"
                        for i in range(len(pca.explained_variance_ratio_))
                    ],
                    "Variance": pca.explained_variance_ratio_,
                    "Cumulative": np.cumsum(
                        pca.explained_variance_ratio_
                    ),
                }
            )

            st.dataframe(
                variance,
                use_container_width=True,
            )

            st.subheader("Components")

            components = pd.DataFrame(
                pca.components_,
                columns=selected,
            )

            st.dataframe(
                components,
                use_container_width=True,
            )

            record(
                "PCA",
                {
                    "explained_variance": (
                        pca.explained_variance_ratio_.tolist()
                    ),
                    "components": pca.components_.tolist(),
                },
            )

# ============================================================
# CLUSTERING
# ============================================================

elif page == "Clustering":

    st.header("K-means clustering")

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.warning("At least two numeric variables are required.")
        st.stop()

    selected = st.multiselect(
        "Variables",
        cols,
        default=cols,
    )

    k = st.slider(
        "Number of clusters",
        2,
        10,
        3,
    )

    if len(selected) >= 2:

        data = df[selected].dropna()

        if len(data) >= k:

            scaled = StandardScaler().fit_transform(data)

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )

            labels = model.fit_predict(scaled)

            result_df = data.copy()

            result_df["Cluster"] = labels

            st.dataframe(
                result_df,
                use_container_width=True,
            )

            st.subheader("Cluster sizes")

            st.write(
                pd.Series(labels).value_counts().sort_index()
            )

            record(
                "Clustering",
                {
                    "k": k,
                    "cluster_sizes": (
                        pd.Series(labels)
                        .value_counts()
                        .to_dict()
                    ),
                },
            )

# ============================================================
# TIME SERIES
# ============================================================

elif page == "Time series":

    st.header("Time series")

    date_columns = [
        col
        for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col])
    ]

    date_columns += [
        col
        for col in df.columns
        if (
            col not in date_columns
            and (
                "date" in col.lower()
                or "time" in col.lower()
            )
        ]

    date_columns = list(dict.fromkeys(date_columns))

    if not date_columns:
        st.warning("No date column detected.")
        st.stop()

    date_col = st.selectbox(
        "Date column",
        date_columns,
    )

    cols = numeric_columns(df)

    if not cols:
        st.warning("No numeric variables detected.")
        st.stop()

    value_col = st.selectbox(
        "Value column",
        cols,
    )

    ts = df[[date_col, value_col]].copy()

    ts[date_col] = pd.to_datetime(
        ts[date_col],
        errors="coerce",
    )

    ts = ts.dropna()

    ts = ts.sort_values(date_col)

    st.dataframe(
        ts,
        use_container_width=True,
    )

    st.line_chart(
        ts.set_index(date_col)[value_col]
    )

    record(
        "Time series",
        {
            "date_column": date_col,
            "value_column": value_col,
            "observations": len(ts),
        },
    )

# ============================================================
# BOOTSTRAP
# ============================================================

elif page == "Bootstrap":

    st.header("Bootstrap confidence intervals")

    cols = numeric_columns(df)

    if not cols:
        st.warning("No numeric variables detected.")
        st.stop()

    selected = st.selectbox(
        "Variable",
        cols,
    )

    iterations = st.slider(
        "Iterations",
        100,
        10000,
        1000,
        step=100,
    )

    data = df[selected].dropna().to_numpy()

    if len(data) >= 2:

        boot_means = []

        for _ in range(iterations):

            sample = np.random.choice(
                data,
                size=len(data),
                replace=True,
            )

            boot_means.append(
                np.mean(sample)
            )

        boot_means = np.asarray(boot_means)

        ci_low, ci_high = np.percentile(
            boot_means,
            [2.5, 97.5],
        )

        st.metric(
            "Mean",
            f"{np.mean(data):.6g}",
        )

        st.write(
            "95% confidence interval:",
            f"[{ci_low:.6g}, {ci_high:.6g}]",
        )

        record(
            "Bootstrap",
            {
                "mean": safe_float(np.mean(data)),
                "CI_lower": safe_float(ci_low),
                "CI_upper": safe_float(ci_high),
                "iterations": iterations,
            },
        )

# ============================================================
# EFFECT SIZE
# ============================================================

elif page == "Effect size":

    st.header("Effect size")

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.warning("At least two numeric variables are required.")
        st.stop()

    a_col = st.selectbox(
        "Group / variable A",
        cols,
    )

    b_col = st.selectbox(
        "Group / variable B",
        [c for c in cols if c != a_col],
    )

    paired = df[[a_col, b_col]].dropna()

    if len(paired) >= 2:

        a = paired[a_col].to_numpy()
        b = paired[b_col].to_numpy()

        mean_a = np.mean(a)
        mean_b = np.mean(b)

        std_a = np.std(a, ddof=1)
        std_b = np.std(b, ddof=1)

        pooled_std = np.sqrt(
            (
                (len(a) - 1) * std_a**2
                + (len(b) - 1) * std_b**2
            )
            / (
                len(a) + len(b) - 2
            )
        )

        if pooled_std != 0:

            d = (
                mean_a - mean_b
            ) / pooled_std

            st.metric(
                "Cohen's d",
                f"{d:.6g}",
            )

            record(
                "Effect size",
                {
                    "Cohen_d": safe_float(d),
                },
            )

# ============================================================
# POWER ANALYSIS
# ============================================================

elif page == "Power analysis":

    st.header("Power analysis")

    st.info(
        "This module estimates statistical power "
        "for a two-group independent t-test."
    )

    from statsmodels.stats.power import TTestIndPower

    effect = st.number_input(
        "Effect size (Cohen's d)",
        value=0.5,
        min_value=0.01,
    )

    alpha = st.number_input(
        "Alpha",
        value=0.05,
        min_value=0.001,
        max_value=0.5,
    )

    sample_size = st.number_input(
        "Sample size per group",
        value=30,
        min_value=2,
    )

    analysis = TTestIndPower()

    power = analysis.power(
        effect_size=effect,
        nobs1=sample_size,
        alpha=alpha,
    )

    st.metric(
        "Estimated power",
        f"{power:.4f}",
    )

    record(
        "Power analysis",
        {
            "effect_size": effect,
            "alpha": alpha,
            "sample_size_per_group": sample_size,
            "power": safe_float(power),
        },
    )

# ============================================================
# PROFILING
# ============================================================

elif page == "Profiling":

    st.header("Dataset profiling")

    profile = pd.DataFrame(
        {
            "Type": df.dtypes.astype(str),
            "Missing": df.isna().sum(),
            "Missing_percent": (
                df.isna().mean() * 100
            ),
            "Unique": df.nunique(),
        }
    )

    st.dataframe(
        profile,
        use_container_width=True,
    )

    st.subheader("Numeric statistics")

    cols = numeric_columns(df)

    if cols:

        st.dataframe(
            df[cols].describe(),
            use_container_width=True,
        )

    st.subheader("Duplicate rows")

    st.metric(
        "Duplicates",
        int(df.duplicated().sum()),
    )

    st.subheader("Memory usage")

    memory = df.memory_usage(
        deep=True
    ).sum()

    st.metric(
        "Memory",
        f"{memory / 1024:.2f} KB",
    )

    record(
        "Profiling",
        {
            "rows": len(df),
            "columns": len(df.columns),
            "missing": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_bytes": int(memory),
        },
    )

# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.header("Analysis report")

    st.subheader("Dataset summary")

    st.write(
        f"Rows: {len(df)}"
    )

    st.write(
        f"Columns: {len(df.columns)}"
    )

    st.write(
        f"Missing values: {int(df.isna().sum().sum())}"
    )

    st.write(
        f"Duplicates: {int(df.duplicated().sum())}"
    )

    st.subheader("Numeric summary")

    cols = numeric_columns(df)

    if cols:

        st.dataframe(
            df[cols].describe(),
            use_container_width=True,
        )

    st.subheader("Previous analyses")

    results = st.session_state.get(
        "analysis_results",
        [],
    )

    if results:

        for item in results:

            with st.expander(
                item["analysis"]
            ):

                st.json(
                    item["results"]
                )

    else:

        st.info(
            "No analyses have been recorded yet."
        )

# ============================================================
# EXPORT
# ============================================================

elif page == "Export":

    st.header("Export")

    st.subheader("Dataset")

    csv_data = df.to_csv(
        index=False,
    ).encode("utf-8")

    st.download_button(
        "Download dataset CSV",
        csv_data,
        file_name="biodatalab_dataset.csv",
        mime="text/csv",
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset",
        )

    st.download_button(
        "Download dataset Excel",
        excel_buffer.getvalue(),
        file_name="biodatalab_dataset.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument"
            ".spreadsheetml.sheet"
        ),
    )

    st.subheader("Analysis results")

    results = st.session_state.get(
        "analysis_results",
        [],
    )

    st.download_button(
        "Download analyses JSON",
        export_results(),
        file_name="biodatalab_results.json",
        mime="application/json",
    )

    st.subheader("Reset")

    if st.button(
        "Clear analysis history",
        type="secondary",
    ):

        st.session_state["analysis_results"] = []

        st.success(
            "Analysis history cleared."
        )

# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "BioDataLab — Open-source scientific analysis"
)
