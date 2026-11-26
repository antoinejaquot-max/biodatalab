
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="BioDataLab",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 BioDataLab")
st.caption("Scientific, biological and agricultural data analysis")

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None

if "original_data" not in st.session_state:
    st.session_state.original_data = None

if "history" not in st.session_state:
    st.session_state.history = []

if "results" not in st.session_state:
    st.session_state.results = {}


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def record(name, result):
    st.session_state.history.append({
        "analysis": name,
        "timestamp": datetime.now().isoformat(),
    })
    st.session_state.results[name] = result


def csv_download(df, filename):
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def json_download(data, filename):
    st.download_button(
        "Download JSON",
        data=json.dumps(data, indent=2, default=str),
        file_name=filename,
        mime="application/json",
    )


def load_file(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(file)
        except Exception:
            file.seek(0)
            return pd.read_csv(file, sep=";")

    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)

    if name.endswith(".json"):
        return pd.read_json(file)

    if name.endswith(".parquet"):
        return pd.read_parquet(file)

    raise ValueError("Unsupported file format.")


# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data",
        "Cleaning",
        "Descriptive statistics",
        "Statistical tests",
        "Regression",
        "Correlation",
        "Sampling",
        "Outliers",
        "PCA",
        "Clustering",
        "Time series",
        "Bootstrap",
        "Effect size",
        "Power analysis",
        "Profiling",
        "Reports",
        "Export",
    ],
)

df = st.session_state.data


# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------

if page == "Overview":

    st.header("Scientific data analysis platform")

    st.markdown("""
    BioDataLab is an open-source Python platform for scientific,
    biological and agricultural data analysis.

    Its analytical tools include:

    - Data cleaning and validation
    - Descriptive statistics
    - Statistical hypothesis testing
    - Regression
    - Correlation
    - Sampling and sampling diagnostics
    - Outlier detection
    - Principal component analysis
    - Clustering
    - Time series analysis
    - Bootstrap resampling
    - Effect size estimation
    - Data profiling
    - Reporting and export
    """)

    if df is not None:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Numeric variables", len(numeric_columns(df)))
        c4.metric("Missing cells", int(df.isna().sum().sum()))

        st.subheader("Current dataset")
        st.dataframe(df.head(10), use_container_width=True)

    else:

        st.info("Upload a dataset in the Data section.")


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

elif page == "Data":

    st.header("Data management")

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    if uploaded is not None:

        try:

            loaded = load_file(uploaded)

            st.session_state.data = loaded.copy()
            st.session_state.original_data = loaded.copy()

            df = loaded

            st.success("Dataset loaded successfully.")

        except Exception as e:

            st.error(str(e))

    df = st.session_state.data

    if df is not None:

        st.subheader("Dataset")

        st.dataframe(df, use_container_width=True)

        st.subheader("Column information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique": df.nunique().values,
        })

        st.dataframe(info, use_container_width=True)

        st.subheader("Preview")

        n = st.slider(
            "Rows to display",
            1,
            max(1, min(len(df), 100)),
            min(10, len(df)),
        )

        st.dataframe(df.head(n), use_container_width=True)

        selected = st.multiselect(
            "Columns",
            df.columns.tolist(),
            default=df.columns.tolist(),
        )

        if selected:
            st.dataframe(df[selected], use_container_width=True)

        if st.button("Restore original dataset"):

            if st.session_state.original_data is not None:

                st.session_state.data = (
                    st.session_state.original_data.copy()
                )

                st.success("Original dataset restored.")


# ------------------------------------------------------------
# CLEANING
# ------------------------------------------------------------

elif page == "Cleaning":

    st.header("Data cleaning")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        st.subheader("Missing values")

        missing = pd.DataFrame({
            "Column": df.columns,
            "Missing": df.isna().sum().values,
            "Percentage": (
                df.isna().sum().values / max(len(df), 1) * 100
            ),
        })

        st.dataframe(missing, use_container_width=True)

        method = st.selectbox(
            "Treatment",
            [
                "Drop rows with missing values",
                "Drop fully empty columns",
                "Fill numeric values with mean",
                "Fill numeric values with median",
                "Fill missing values with zero",
                "Fill with custom value",
            ],
        )

        custom = None

        if method == "Fill with custom value":
            custom = st.text_input("Custom replacement")

        if st.button("Apply treatment"):

            new = df.copy()

            try:

                if method == "Drop rows with missing values":
                    new = new.dropna()

                elif method == "Drop fully empty columns":
                    new = new.dropna(axis=1, how="all")

                elif method == "Fill numeric values with mean":

                    for col in numeric_columns(new):
                        new[col] = new[col].fillna(new[col].mean())

                elif method == "Fill numeric values with median":

                    for col in numeric_columns(new):
                        new[col] = new[col].fillna(new[col].median())

                elif method == "Fill missing values with zero":
                    new = new.fillna(0)

                elif method == "Fill with custom value":
                    new = new.fillna(custom)

                st.session_state.data = new

                record("Missing value treatment", method)

                st.success("Treatment applied.")

            except Exception as e:

                st.error(str(e))

        st.divider()

        st.subheader("Duplicate rows")

        st.metric("Duplicates", int(df.duplicated().sum()))

        if st.button("Remove duplicates"):

            st.session_state.data = df.drop_duplicates()

            st.success("Duplicates removed.")

        st.divider()

        st.subheader("Remove columns")

        remove = st.multiselect(
            "Columns to remove",
            df.columns.tolist(),
        )

        if st.button("Remove selected columns"):

            st.session_state.data = df.drop(columns=remove)

            st.success("Columns removed.")

        st.divider()

        st.subheader("Convert columns to numeric")

        convert = st.multiselect(
            "Columns to convert",
            df.columns.tolist(),
        )

        if st.button("Convert"):

            new = df.copy()

            for col in convert:
                new[col] = pd.to_numeric(new[col], errors="coerce")

            st.session_state.data = new

            st.success("Conversion completed.")

        st.subheader("Current dataset")

        st.dataframe(st.session_state.data, use_container_width=True)


# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

elif page == "Descriptive statistics":

    st.header("Descriptive statistics")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            if selected:

                result = df[selected].describe().T

                result["Variance"] = df[selected].var()
                result["Skewness"] = df[selected].skew()
                result["Kurtosis"] = df[selected].kurtosis()

                st.dataframe(result, use_container_width=True)

                csv_download(
                    result.reset_index(),
                    "descriptive_statistics.csv",
                )

                col = st.selectbox("Distribution", selected)

                st.histogram(df[col].dropna())

                st.subheader("Quantiles")

                q = df[selected].quantile([
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ])

                st.dataframe(q, use_container_width=True)

        else:

            st.warning("No numeric variables found.")


# ------------------------------------------------------------
# STATISTICAL TESTS
# ------------------------------------------------------------

elif page == "Statistical tests":

    st.header("Statistical tests")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)
        cats = categorical_columns(df)

        if nums:

            test = st.selectbox(
                "Test",
                [
                    "One-sample t-test",
                    "Independent t-test",
                    "Paired t-test",
                    "Mann-Whitney",
                    "Wilcoxon",
                    "One-way ANOVA",
                    "Kruskal-Wallis",
                    "Shapiro-Wilk",
                ],
            )

            alpha = st.slider(
                "Alpha",
                0.01,
                0.10,
                0.05,
            )

            if test == "One-sample t-test":

                col = st.selectbox("Variable", nums)

                mu = st.number_input("Reference mean", value=0.0)

                if st.button("Run test"):

                    x = clean_numeric(df[col])

                    result = stats.ttest_1samp(x, mu)

                    st.write(result)

                    st.metric("p-value", result.pvalue)

                    st.write(
                        "Reject H0"
                        if result.pvalue < alpha
                        else "Do not reject H0"
                    )

            elif test == "Independent t-test":

                if cats:

                    col = st.selectbox("Numeric variable", nums)
                    group = st.selectbox("Grouping variable", cats)

                    levels = df[group].dropna().unique().tolist()

                    if len(levels) >= 2:

                        g1 = st.selectbox("Group 1", levels)

                        g2 = st.selectbox(
                            "Group 2",
                            [x for x in levels if x != g1],
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

                            st.write(result)
                            st.metric("p-value", result.pvalue)

            elif test == "Paired t-test":

                col1 = st.selectbox("Variable 1", nums)

                col2 = st.selectbox(
                    "Variable 2",
                    [x for x in nums if x != col1],
                )

                if st.button("Run test"):

                    paired = df[[col1, col2]].dropna()

                    result = stats.ttest_rel(
                        paired[col1],
                        paired[col2],
                    )

                    st.write(result)
                    st.metric("p-value", result.pvalue)

            elif test == "Mann-Whitney":

                if cats:

                    col = st.selectbox("Variable", nums)
                    group = st.selectbox("Grouping variable", cats)

                    levels = df[group].dropna().unique().tolist()

                    if len(levels) >= 2:

                        g1 = st.selectbox("Group 1", levels)

                        g2 = st.selectbox(
                            "Group 2",
                            [x for x in levels if x != g1],
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

                            st.write(result)
                            st.metric("p-value", result.pvalue)

            elif test == "Wilcoxon":

                col1 = st.selectbox("Variable 1", nums)

                col2 = st.selectbox(
                    "Variable 2",
                    [x for x in nums if x != col1],
                )

                if st.button("Run test"):

                    paired = df[[col1, col2]].dropna()

                    result = stats.wilcoxon(
                        paired[col1],
                        paired[col2],
                    )

                    st.write(result)
                    st.metric("p-value", result.pvalue)

            elif test == "One-way ANOVA":

                if cats:

                    col = st.selectbox("Variable", nums)
                    group = st.selectbox("Grouping variable", cats)

                    levels = df[group].dropna().unique()

                    data = [
                        clean_numeric(df.loc[df[group] == level, col])
                        for level in levels
                    ]

                    data = [x for x in data if len(x) > 0]

                    if st.button("Run test"):

                        if len(data) >= 2:

                            result = stats.f_oneway(*data)

                            st.write(result)
                            st.metric("p-value", result.pvalue)

            elif test == "Kruskal-Wallis":

                if cats:

                    col = st.selectbox("Variable", nums)
                    group = st.selectbox("Grouping variable", cats)

                    levels = df[group].dropna().unique()

                    data = [
                        clean_numeric(df.loc[df[group] == level, col])
                        for level in levels
                    ]

                    data = [x for x in data if len(x) > 0]

                    if st.button("Run test"):

                        if len(data) >= 2:

                            result = stats.kruskal(*data)

                            st.write(result)
                            st.metric("p-value", result.pvalue)

            elif test == "Shapiro-Wilk":

                col = st.selectbox("Variable", nums)

                if st.button("Run test"):

                    x = clean_numeric(df[col])

                    if 3 <= len(x) <= 5000:

                        result = stats.shapiro(x)

                        st.write(result)
                        st.metric("p-value", result.pvalue)

                    else:

                        st.warning(
                            "Shapiro-Wilk requires 3 to 5000 observations."
                        )


# ------------------------------------------------------------
# REGRESSION
# ------------------------------------------------------------

elif page == "Regression":

    st.header("Linear regression")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            target = st.selectbox("Target", nums)

            features = st.multiselect(
                "Predictors",
                [x for x in nums if x != target],
            )

            test_size = st.slider(
                "Test size",
                0.1,
                0.5,
                0.2,
            )

            if features and st.button("Fit regression"):

                data = df[features + [target]].dropna()

                X = data[features]
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                )

                model = LinearRegression()

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                st.metric(
                    "R²",
                    r2_score(y_test, predictions),
                )

                st.metric(
                    "RMSE",
                    np.sqrt(
                        mean_squared_error(y_test, predictions)
                    ),
                )

                coef = pd.DataFrame({
                    "Variable": features,
                    "Coefficient": model.coef_,
                })

                st.dataframe(coef, use_container_width=True)

                result = pd.DataFrame({
                    "Observed": y_test.values,
                    "Predicted": predictions,
                })

                st.dataframe(result, use_container_width=True)

                csv_download(
                    result,
                    "regression_predictions.csv",
                )

                record(
                    "Regression",
                    {
                        "R2": r2_score(y_test, predictions),
                        "RMSE": np.sqrt(
                            mean_squared_error(y_test, predictions)
                        ),
                    },
                )

        else:

            st.warning("At least two numeric columns are required.")


# ------------------------------------------------------------
# CORRELATION
# ------------------------------------------------------------

elif page == "Correlation":

    st.header("Correlation analysis")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if len(nums) >= 2:

            method = st.selectbox(
                "Method",
                ["pearson", "spearman", "kendall"],
            )

            selected = st.multiselect(
                "Variables",
                nums,
                default=nums,
            )

            if len(selected) >= 2:

                corr = df[selected].corr(method=method)

                st.dataframe(corr, use_container_width=True)

                csv_download(
                    corr.reset_index(),
                    "correlation.csv",
                )

                pairs = []

                for i, col1 in enumerate(selected):

                    for col2 in selected[i + 1:]:

                        data = df[[col1, col2]].dropna()

                        if len(data) >= 2:

                            if method == "pearson":
                                result = stats.pearsonr(
                                    data[col1],
                                    data[col2],
                                )

                            elif method == "spearman":
                                result = stats.spearmanr(
                                    data[col1],
                                    data[col2],
                                )

                            else:
                                result = stats.kendalltau(
                                    data[col1],
                                    data[col2],
                                )

                            pairs.append({
                                "Variable 1": col1,
                                "Variable 2": col2,
                                "Correlation": result.statistic,
                                "p-value": result.pvalue,
                            })

                st.subheader("Pairwise results")

                st.dataframe(
                    pd.DataFrame(pairs),
                    use_container_width=True,
                )


# ------------------------------------------------------------
# SAMPLING
# ------------------------------------------------------------

elif page == "Sampling":

    st.header("Sampling")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        method = st.selectbox(
            "Method",
            ["Random sampling", "Stratified sampling"],
        )

        size = st.number_input(
            "Sample size",
            min_value=1,
            max_value=len(df),
            value=max(1, min(len(df), int(len(df) * 0.2))),
        )

        cats = categorical_columns(df)

        stratify_column = None

        if method == "Stratified sampling" and cats:

            stratify_column = st.selectbox(
                "Stratification variable",
                cats,
            )

        if method == "Stratified sampling" and not cats:

            st.warning("Categorical variable required.")

        if st.button("Generate sample"):

            try:

                if method == "Random sampling":

                    sample = df.sample(
                        n=int(size),
                        random_state=42,
                    )

                else:

                    data = df.dropna(
                        subset=[stratify_column]
                    )

                    proportions = (
                        data[stratify_column]
                        .value_counts(normalize=True)
                    )

                    parts = []

                    for level, proportion in proportions.items():

                        group = data[
                            data[stratify_column] == level
                        ]

                        n = max(
                            1,
                            round(int(size) * proportion),
                        )

                        n = min(n, len(group))

                        parts.append(
                            group.sample(
                                n=n,
                                random_state=42,
                            )
                        )

                    sample = pd.concat(parts)

                    if len(sample) > int(size):

                        sample = sample.sample(
                            n=int(size),
                            random_state=42,
                        )

                st.dataframe(sample, use_container_width=True)

                st.metric("Sample size", len(sample))

                csv_download(
                    sample,
                    "sample.csv",
                )

                if method == "Stratified sampling":

                    st.subheader("Population proportions")

                    st.dataframe(
                        df[stratify_column].value_counts(
                            normalize=True
                        ),
                    )

                    st.subheader("Sample proportions")

                    st.dataframe(
                        sample[stratify_column].value_counts(
                            normalize=True
                        ),
                    )

                record(
                    "Sampling",
                    {
                        "method": method,
                        "size": len(sample),
                    },
                )

            except Exception as e:

                st.error(str(e))


# ------------------------------------------------------------
# OUTLIERS
# ------------------------------------------------------------

elif page == "Outliers":

    st.header("Outlier detection")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox("Variable", nums)

            method = st.selectbox(
                "Method",
                ["IQR", "Z-score"],
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

            st.metric("Outliers detected", len(outliers))

            st.dataframe(
                outliers,
                use_container_width=True,
            )

            csv_download(
                outliers,
                "outliers.csv",
            )

            if st.button("Remove outliers"):

                st.session_state.data = df[~mask].copy()

                st.success("Outliers removed.")


# ------------------------------------------------------------
# PCA
# ------------------------------------------------------------

elif page == "PCA":

    st.header("Principal component analysis")

    df = st.session_state.data

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

            if len(selected) >= 2:

                n_components = st.slider(
                    "Number of components",
                    2,
                    min(10, len(selected)),
                    2,
                )

                if st.button("Run PCA"):

                    data = df[selected].dropna()

                    scaled = StandardScaler().fit_transform(data)

                    model = PCA(
                        n_components=n_components,
                    )

                    transformed = model.fit_transform(scaled)

                    result = pd.DataFrame(
                        transformed,
                        columns=[
                            f"PC{i+1}"
                            for i in range(n_components)
                        ],
                    )

                    st.subheader("Transformed data")

                    st.dataframe(
                        result,
                        use_container_width=True,
                    )

                    variance = pd.DataFrame({
                        "Component": [
                            f"PC{i+1}"
                            for i in range(n_components)
                        ],
                        "Explained variance": (
                            model.explained_variance_ratio_
                        ),
                        "Cumulative variance": (
                            np.cumsum(
                                model.explained_variance_ratio_
                            )
                        ),
                    })

                    st.subheader("Explained variance")

                    st.dataframe(
                        variance,
                        use_container_width=True,
                    )

                    loadings = pd.DataFrame(
                        model.components_.T,
                        index=selected,
                        columns=[
                            f"PC{i+1}"
                            for i in range(n_components)
                        ],
                    )

                    st.subheader("Loadings")

                    st.dataframe(
                        loadings,
                        use_container_width=True,
                    )

                    csv_download(
                        result,
                        "pca_results.csv",
                    )

                    csv_download(
                        loadings.reset_index(),
                        "pca_loadings.csv",
                    )


# ------------------------------------------------------------
# CLUSTERING
# ------------------------------------------------------------

elif page == "Clustering":

    st.header("K-means clustering")

    df = st.session_state.data

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
                "Clusters",
                2,
                10,
                3,
            )

            if len(selected) >= 2:

                if st.button("Run clustering"):

                    data = df[selected].dropna()

                    scaled = StandardScaler().fit_transform(data)

                    model = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=10,
                    )

                    labels = model.fit_predict(scaled)

                    result = data.copy()

                    result["Cluster"] = labels

                    st.dataframe(
                        result,
                        use_container_width=True,
                    )

                    st.subheader("Cluster sizes")

                    st.dataframe(
                        result["Cluster"].value_counts().sort_index(),
                    )

                    st.subheader("Cluster centers")

                    centers = pd.DataFrame(
                        model.cluster_centers_,
                        columns=selected,
                    )

                    st.dataframe(
                        centers,
                        use_container_width=True,
                    )

                    st.metric("Inertia", model.inertia_)

                    csv_download(
                        result,
                        "clusters.csv",
                    )


# ------------------------------------------------------------
# TIME SERIES
# ------------------------------------------------------------

elif page == "Time series":

    st.header("Time series analysis")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

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

            if st.button("Analyze"):

                data = df[[date_col, value_col]].copy()

                data[date_col] = pd.to_datetime(
                    data[date_col],
                    errors="coerce",
                )

                data[value_col] = pd.to_numeric(
                    data[value_col],
                    errors="coerce",
                )

                data = data.dropna().sort_values(date_col)

                data = data.set_index(date_col)

                st.subheader("Time series")

                st.line_chart(data[value_col])

                window = st.slider(
                    "Rolling window",
                    2,
                    max(2, min(100, len(data))),
                    min(7, max(2, len(data))),
                )

                rolling_mean = data[value_col].rolling(
                    window
                ).mean()

                rolling_std = data[value_col].rolling(
                    window
                ).std()

                st.subheader("Rolling mean")

                st.line_chart(rolling_mean)

                st.subheader("Rolling standard deviation")

                st.line_chart(rolling_std)

                result = pd.DataFrame({
                    "Original": data[value_col],
                    "Rolling mean": rolling_mean,
                    "Rolling standard deviation": rolling_std,
                })

                csv_download(
                    result.reset_index(),
                    "time_series.csv",
                )

        else:

            st.warning(
                "Date and numeric columns are required."
            )


# ------------------------------------------------------------
# BOOTSTRAP
# ------------------------------------------------------------

elif page == "Bootstrap":

    st.header("Bootstrap")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox("Variable", nums)

            iterations = st.number_input(
                "Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
            )

            confidence = st.selectbox(
                "Confidence level",
                [90, 95, 99],
                index=1,
            )

            statistic = st.selectbox(
                "Statistic",
                ["Mean", "Median"],
            )

            if st.button("Run bootstrap"):

                x = clean_numeric(df[col]).values

                if len(x):

                    results = []

                    for _ in range(int(iterations)):

                        sample = np.random.choice(
                            x,
                            size=len(x),
                            replace=True,
                        )

                        if statistic == "Mean":
                            results.append(np.mean(sample))

                        else:
                            results.append(np.median(sample))

                    results = np.asarray(results)

                    alpha = 1 - confidence / 100

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
                        float(np.mean(results)),
                    )

                    st.metric("Lower bound", float(lower))
                    st.metric("Upper bound", float(upper))

                    result = pd.DataFrame({
                        "Bootstrap statistic": results,
                    })

                    st.dataframe(
                        result.head(100),
                        use_container_width=True,
                    )

                    csv_download(
                        result,
                        "bootstrap.csv",
                    )


# ------------------------------------------------------------
# EFFECT SIZE
# ------------------------------------------------------------

elif page == "Effect size":

    st.header("Effect size")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)
        cats = categorical_columns(df)

        if nums and cats:

            col = st.selectbox("Numeric variable", nums)
            group = st.selectbox("Grouping variable", cats)

            levels = df[group].dropna().unique().tolist()

            if len(levels) >= 2:

                g1 = st.selectbox("Group 1", levels)

                g2 = st.selectbox(
                    "Group 2",
                    [x for x in levels if x != g1],
                )

                if st.button("Calculate Cohen's d"):

                    x = clean_numeric(
                        df.loc[df[group] == g1, col]
                    )

                    y = clean_numeric(
                        df.loc[df[group] == g2, col]
                    )

                    if len(x) > 1 and len(y) > 1:

                        pooled = np.sqrt(
                            (
                                (len(x) - 1) * x.var()
                                + (len(y) - 1) * y.var()
                            )
                            / (len(x) + len(y) - 2)
                        )

                        if pooled > 0:

                            d = (
                                x.mean() - y.mean()
                            ) / pooled

                            st.metric(
                                "Cohen's d",
                                float(d),
                            )

                        else:

                            st.warning(
                                "Pooled standard deviation is zero."
                            )


# ------------------------------------------------------------
# POWER ANALYSIS
# ------------------------------------------------------------

elif page == "Power analysis":

    st.header("Statistical power context")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        nums = numeric_columns(df)

        if nums:

            col = st.selectbox("Variable", nums)

            mu = st.number_input(
                "Reference mean",
                value=0.0,
            )

            x = clean_numeric(df[col])

            if len(x) > 1:

                st.metric("Sample size", len(x))
                st.metric("Mean", float(x.mean()))
                st.metric("Standard deviation", float(x.std()))

                if x.std() > 0:

                    effect = (
                        x.mean() - mu
                    ) / x.std()

                    st.metric(
                        "Observed standardized effect",
                        float(effect),
                    )

                st.info(
                    "Observed effect size is descriptive. "
                    "Post-hoc power based on observed effect size "
                    "does not replace prospective power analysis."
                )


# ------------------------------------------------------------
# PROFILING
# ------------------------------------------------------------

elif page == "Profiling":

    st.header("Data profiling")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        rows = []

        for col in df.columns:

            s = df[col]

            rows.append({
                "Column": col,
                "Type": str(s.dtype),
                "Rows": len(s),
                "Missing": int(s.isna().sum()),
                "Missing percentage": (
                    s.isna().sum() / max(len(s), 1) * 100
                ),
                "Unique": int(s.nunique()),
                "Memory bytes": int(
                    s.memory_usage(deep=True)
                ),
            })

        profile = pd.DataFrame(rows)

        st.dataframe(
            profile,
            use_container_width=True,
        )

        st.write({
            "Rows": len(df),
            "Columns": len(df.columns),
            "Duplicates": int(df.duplicated().sum()),
            "Missing cells": int(df.isna().sum().sum()),
            "Numeric variables": len(numeric_columns(df)),
            "Categorical variables": len(categorical_columns(df)),
        })

        csv_download(
            profile,
            "profile.csv",
        )


# ------------------------------------------------------------
# REPORTS
# ------------------------------------------------------------

elif page == "Reports":

    st.header("Reports")

    df = st.session_state.data

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
            "duplicates": int(df.duplicated().sum()),
            "descriptive_statistics": df.describe().to_dict(),
            "analysis_history": st.session_state.history,
        }

        st.subheader("Summary")

        st.write({
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing": int(df.isna().sum().sum()),
            "Duplicates": int(df.duplicated().sum()),
        })

        st.subheader("Descriptive statistics")

        st.dataframe(
            df.describe(),
            use_container_width=True,
        )

        st.subheader("Analysis history")

        if st.session_state.history:

            st.dataframe(
                pd.DataFrame(st.session_state.history),
                use_container_width=True,
            )

        else:

            st.info("No analysis recorded yet.")

        json_download(
            report,
            "biodatalab_report.json",
        )


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

elif page == "Export":

    st.header("Export")

    df = st.session_state.data

    if df is None:

        st.warning("Upload a dataset first.")

    else:

        st.dataframe(
            df,
            use_container_width=True,
        )

        csv_download(
            df,
            "biodatalab_dataset.csv",
        )

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
            "Download Excel",
            data=excel_buffer.getvalue(),
            file_name="biodatalab.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

        if st.session_state.results:

            json_download(
                st.session_state.results,
                "analysis_results.json",
            )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "BioDataLab | Open-source scientific data analysis"
)
