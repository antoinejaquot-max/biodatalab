
from __future__ import annotations
import io, json
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

st.set_page_config(page_title="BioDataLab", page_icon="🧬", layout="wide")
st.title("🧬 BioDataLab")
st.caption("Open-source scientific, biological and agricultural data analysis")

for key, default in [("data", None), ("original_data", None), ("history", []), ("results", {})]:
    if key not in st.session_state:
        st.session_state[key] = default


def numeric_columns(frame):
    return frame.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(frame):
    return frame.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def record(name, result=None):
    st.session_state.history.append({"analysis": name, "timestamp": datetime.now().isoformat()})
    if result is not None:
        st.session_state.results[name] = result


def download_csv(frame, filename):
    st.download_button("Download CSV", frame.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def download_json(data, filename):
    st.download_button("Download JSON", json.dumps(data, indent=2, default=str), filename, "application/json")


def load_file(upload):
    filename = upload.name.lower()
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(upload)
        except Exception:
            upload.seek(0)
            return pd.read_csv(upload, sep=";")
    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(upload)
    if filename.endswith(".json"):
        return pd.read_json(upload)
    if filename.endswith(".parquet"):
        return pd.read_parquet(upload)
    raise ValueError("Unsupported file format")


page = st.sidebar.radio("Navigation", [
    "Overview", "Data", "Cleaning", "Descriptive statistics", "Statistical tests",
    "Regression", "Correlation", "Sampling", "Outliers", "PCA", "Clustering",
    "Time series", "Bootstrap", "Effect size", "Power analysis", "Profiling", "Reports", "Export",
])
df = st.session_state.data

if page == "Overview":
    st.header("Scientific data analysis platform")
    st.markdown("BioDataLab provides tools for data preparation, statistical analysis, regression, sampling, multivariate analysis, visualization and reporting.")
    st.markdown("**Available domains:** cleaning and validation, descriptive statistics, hypothesis testing, regression, correlation, sampling, outliers, PCA, clustering, time series, bootstrap, effect sizes, profiling and export.")
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(df)); c2.metric("Columns", len(df.columns)); c3.metric("Numeric variables", len(numeric_columns(df))); c4.metric("Missing cells", int(df.isna().sum().sum()))
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a dataset from the Data section to begin.")

elif page == "Data":
    st.header("Data management")
    upload = st.file_uploader("Upload data", type=["csv", "xlsx", "xls", "json", "parquet"])
    if upload is not None:
        try:
            loaded = load_file(upload)
            st.session_state.data = loaded.copy(); st.session_state.original_data = loaded.copy(); df = loaded
            st.success("Dataset loaded.")
        except Exception as exc:
            st.error(str(exc))
    df = st.session_state.data
    if df is not None:
        st.dataframe(df, use_container_width=True)
        info = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Missing": df.isna().sum().values, "Unique": df.nunique().values})
        st.subheader("Column information"); st.dataframe(info, use_container_width=True)
        n = st.slider("Preview rows", min_value=1, max_value=max(1, min(len(df), 100)), value=min(10, len(df)))
        st.dataframe(df.head(n), use_container_width=True)
        chosen = st.multiselect("Display columns", df.columns.tolist(), default=df.columns.tolist())
        if chosen: st.dataframe(df[chosen], use_container_width=True)
        if st.button("Restore original dataset") and st.session_state.original_data is not None:
            st.session_state.data = st.session_state.original_data.copy(); st.success("Restored.")

elif page == "Cleaning":
    st.header("Data cleaning")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        missing = pd.DataFrame({"Column": df.columns, "Missing": df.isna().sum().values, "Percent": df.isna().sum().values / max(len(df), 1) * 100})
        st.subheader("Missing values"); st.dataframe(missing, use_container_width=True)
        action = st.selectbox("Missing-value treatment", ["Drop rows with missing values", "Drop fully empty columns", "Fill numeric values with mean", "Fill numeric values with median", "Fill missing values with zero", "Fill with custom value"])
        custom = st.text_input("Replacement value") if action == "Fill with custom value" else None
        if st.button("Apply missing-value treatment"):
            new = df.copy()
            if action == "Drop rows with missing values": new = new.dropna()
            elif action == "Drop fully empty columns": new = new.dropna(axis=1, how="all")
            elif action == "Fill numeric values with mean":
                for col in numeric_columns(new): new[col] = new[col].fillna(new[col].mean())
            elif action == "Fill numeric values with median":
                for col in numeric_columns(new): new[col] = new[col].fillna(new[col].median())
            elif action == "Fill missing values with zero": new = new.fillna(0)
            elif action == "Fill with custom value": new = new.fillna(custom)
            st.session_state.data = new; record("Missing-value treatment", action); st.success("Treatment applied.")
        st.divider(); st.subheader("Duplicates"); st.metric("Duplicate rows", int(df.duplicated().sum()))
        if st.button("Remove duplicate rows"): st.session_state.data = df.drop_duplicates(); st.success("Duplicates removed.")
        cols = st.multiselect("Remove columns", df.columns.tolist())
        if st.button("Remove selected columns"): st.session_state.data = df.drop(columns=cols); st.success("Columns removed.")
        cols_convert = st.multiselect("Convert to numeric", df.columns.tolist())
        if st.button("Convert selected columns"):
            new = df.copy()
            for col in cols_convert: new[col] = pd.to_numeric(new[col], errors="coerce")
            st.session_state.data = new; st.success("Conversion complete.")
        st.subheader("Current dataset"); st.dataframe(st.session_state.data, use_container_width=True)

elif page == "Descriptive statistics":
    st.header("Descriptive statistics")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if nums:
            selected = st.multiselect("Variables", nums, default=nums)
            if selected:
                result = df[selected].describe().T; result["Variance"] = df[selected].var(); result["Skewness"] = df[selected].skew(); result["Kurtosis"] = df[selected].kurtosis()
                st.dataframe(result, use_container_width=True); download_csv(result.reset_index(), "descriptive_statistics.csv")
                col = st.selectbox("Histogram variable", selected); st.histogram(df[col].dropna())
                st.subheader("Quantiles"); st.dataframe(df[selected].quantile([.01, .05, .25, .5, .75, .95, .99]), use_container_width=True)
        else: st.warning("No numeric columns.")

elif page == "Statistical tests":
    st.header("Statistical tests")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums, cats = numeric_columns(df), categorical_columns(df)
        if nums:
            test = st.selectbox("Test", ["One-sample t-test", "Independent t-test", "Paired t-test", "Mann-Whitney", "Wilcoxon", "One-way ANOVA", "Kruskal-Wallis", "Shapiro-Wilk"])
            alpha = st.slider("Significance level", .01, .10, .05)
            if test == "One-sample t-test":
                col = st.selectbox("Variable", nums); mu = st.number_input("Reference mean", value=0.0)
                if st.button("Run test"):
                    res = stats.ttest_1samp(clean_numeric(df[col]), mu); st.write(res); st.write("Reject H0" if res.pvalue < alpha else "Do not reject H0")
            elif test in ["Independent t-test", "Mann-Whitney"]:
                if cats:
                    col = st.selectbox("Numeric variable", nums); group = st.selectbox("Grouping variable", cats); levels = df[group].dropna().unique().tolist()
                    if len(levels) >= 2:
                        g1 = st.selectbox("Group 1", levels); g2 = st.selectbox("Group 2", [x for x in levels if x != g1])
                        if st.button("Run test"):
                            x = clean_numeric(df.loc[df[group] == g1, col]); y = clean_numeric(df.loc[df[group] == g2, col])
                            res = stats.ttest_ind(x, y, equal_var=False) if test == "Independent t-test" else stats.mannwhitneyu(x, y, alternative="two-sided")
                            st.write(res); st.metric("p-value", res.pvalue)
                else: st.warning("A categorical variable is required.")
            elif test in ["Paired t-test", "Wilcoxon"]:
                col1 = st.selectbox("First variable", nums); col2 = st.selectbox("Second variable", [x for x in nums if x != col1])
                if st.button("Run test"):
                    paired = df[[col1, col2]].dropna(); res = stats.ttest_rel(paired[col1], paired[col2]) if test == "Paired t-test" else stats.wilcoxon(paired[col1], paired[col2]); st.write(res); st.metric("p-value", res.pvalue)
            elif test in ["One-way ANOVA", "Kruskal-Wallis"]:
                if cats:
                    col = st.selectbox("Numeric variable", nums); group = st.selectbox("Grouping variable", cats); levels = df[group].dropna().unique()
                    groups = [clean_numeric(df.loc[df[group] == lev, col]) for lev in levels]; groups = [g for g in groups if len(g)]
                    if st.button("Run test") and len(groups) >= 2:
                        res = stats.f_oneway(*groups) if test == "One-way ANOVA" else stats.kruskal(*groups); st.write(res); st.metric("p-value", res.pvalue)
                else: st.warning("A categorical variable is required.")
            elif test == "Shapiro-Wilk":
                col = st.selectbox("Variable", nums)
                if st.button("Run test"):
                    x = clean_numeric(df[col])
                    if 3 <= len(x) <= 5000: res = stats.shapiro(x); st.write(res); st.metric("p-value", res.pvalue)
                    else: st.warning("Shapiro-Wilk requires 3 to 5000 observations.")

elif page == "Regression":
    st.header("Linear regression")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if len(nums) >= 2:
            target = st.selectbox("Target", nums); features = st.multiselect("Predictors", [c for c in nums if c != target]); split = st.slider("Test proportion", .1, .5, .2)
            if features and st.button("Fit model"):
                data = df[features + [target]].dropna(); Xtr, Xte, ytr, yte = train_test_split(data[features], data[target], test_size=split, random_state=42)
                model = LinearRegression().fit(Xtr, ytr); pred = model.predict(Xte)
                st.metric("R²", r2_score(yte, pred)); st.metric("RMSE", np.sqrt(mean_squared_error(yte, pred)))
                st.dataframe(pd.DataFrame({"Variable": features, "Coefficient": model.coef_}), use_container_width=True)
                result = pd.DataFrame({"Observed": yte.values, "Predicted": pred}); st.dataframe(result, use_container_width=True); download_csv(result, "regression_predictions.csv")
                record("Regression", {"R2": r2_score(yte, pred), "RMSE": np.sqrt(mean_squared_error(yte, pred)))
        else: st.warning("At least two numeric variables required.")

elif page == "Correlation":
    st.header("Correlation")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if len(nums) >= 2:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"]); selected = st.multiselect("Variables", nums, default=nums)
            if len(selected) >= 2:
                corr = df[selected].corr(method=method); st.dataframe(corr, use_container_width=True); download_csv(corr.reset_index(), "correlation.csv")
                pairs = []
                for i, c1 in enumerate(selected):
                    for c2 in selected[i + 1:]:
                        pair = df[[c1, c2]].dropna()
                        if len(pair) >= 2:
                            res = stats.pearsonr(pair[c1], pair[c2]) if method == "pearson" else stats.spearmanr(pair[c1], pair[c2]) if method == "spearman" else stats.kendalltau(pair[c1], pair[c2])
                            pairs.append({"Variable 1": c1, "Variable 2": c2, "Correlation": res.statistic, "p-value": res.pvalue})
                st.subheader("Pairwise tests"); st.dataframe(pd.DataFrame(pairs), use_container_width=True)

elif page == "Sampling":
    st.header("Sampling")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        method = st.selectbox("Method", ["Random", "Stratified"]); size = st.number_input("Sample size", 1, len(df), max(1, min(len(df), int(len(df) * .2))))
        cats = categorical_columns(df); strat_col = st.selectbox("Stratification column", cats) if method == "Stratified" and cats else None
        if method == "Stratified" and not cats: st.warning("Categorical column required.")
        if st.button("Generate sample"):
            if method == "Random": sample = df.sample(n=int(size), random_state=42)
            elif strat_col:
                valid = df.dropna(subset=[strat_col]); props = valid[strat_col].value_counts(normalize=True); parts = []
                for lev, prop in props.items():
                    group = valid[valid[strat_col] == lev]; n = min(len(group), max(1, round(int(size) * prop))); parts.append(group.sample(n=n, random_state=42))
                sample = pd.concat(parts)
                if len(sample) > int(size): sample = sample.sample(n=int(size), random_state=42)
            else: sample = pd.DataFrame()
            st.dataframe(sample, use_container_width=True); st.metric("Sample rows", len(sample)); download_csv(sample, "sample.csv")
            if method == "Stratified" and strat_col:
                st.write("Population proportions"); st.dataframe(df[strat_col].value_counts(normalize=True)); st.write("Sample proportions"); st.dataframe(sample[strat_col].value_counts(normalize=True))
            record("Sampling", {"method": method, "size": len(sample)})

elif page == "Outliers":
    st.header("Outlier detection")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if nums:
            col = st.selectbox("Variable", nums); method = st.selectbox("Method", ["IQR", "Z-score"]); x = clean_numeric(df[col])
            if method == "IQR":
                q1, q3 = x.quantile(.25), x.quantile(.75); spread = q3 - q1; low, high = q1 - 1.5 * spread, q3 + 1.5 * spread; mask = (df[col] < low) | (df[col] > high)
            else:
                sd = df[col].std(); mask = (df[col] - df[col].mean()).abs() / sd > 3 if sd != 0 else pd.Series(False, index=df.index)
            out = df[mask]; st.metric("Detected", len(out)); st.dataframe(out, use_container_width=True); download_csv(out, "outliers.csv")
            if st.button("Remove outliers"): st.session_state.data = df[~mask].copy(); st.success("Removed.")

elif page == "PCA":
    st.header("Principal component analysis")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if len(nums) >= 2:
            selected = st.multiselect("Variables", nums, default=nums)
            if len(selected) >= 2:
                ncomp = st.slider("Components", 2, min(10, len(selected)), 2)
                if st.button("Run PCA"):
                    data = df[selected].dropna(); scaled = StandardScaler().fit_transform(data); model = PCA(n_components=ncomp); transformed = model.fit_transform(scaled)
                    result = pd.DataFrame(transformed, columns=[f"PC{i+1}" for i in range(ncomp)]); st.subheader("Scores"); st.dataframe(result, use_container_width=True)
                    variance = pd.DataFrame({"Component": [f"PC{i+1}" for i in range(ncomp)], "Explained variance": model.explained_variance_ratio_, "Cumulative": np.cumsum(model.explained_variance_ratio_)}); st.dataframe(variance, use_container_width=True)
                    loadings = pd.DataFrame(model.components_.T, index=selected, columns=[f"PC{i+1}" for i in range(ncomp)]); st.subheader("Loadings"); st.dataframe(loadings, use_container_width=True); download_csv(result, "pca_scores.csv"); download_csv(loadings.reset_index(), "pca_loadings.csv")

elif page == "Clustering":
    st.header("K-means clustering")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if len(nums) >= 2:
            selected = st.multiselect("Variables", nums, default=nums); k = st.slider("Clusters", 2, 10, 3)
            if len(selected) >= 2 and st.button("Run clustering"):
                data = df[selected].dropna(); scaled = StandardScaler().fit_transform(data); model = KMeans(n_clusters=k, random_state=42, n_init=10); labels = model.fit_predict(scaled); result = data.copy(); result["Cluster"] = labels
                st.dataframe(result, use_container_width=True); st.subheader("Cluster sizes"); st.dataframe(result["Cluster"].value_counts().sort_index()); st.subheader("Centers (scaled space)"); st.dataframe(pd.DataFrame(model.cluster_centers_, columns=selected)); st.metric("Inertia", model.inertia_); download_csv(result, "clusters.csv")

elif page == "Time series":
    st.header("Time series analysis")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if nums:
            date_col = st.selectbox("Date/time column", df.columns.tolist())
            value_col = st.selectbox("Numeric variable", nums)
            if st.button("Analyze time series"):
                data = df[[date_col, value_col]].copy(); data[date_col] = pd.to_datetime(data[date_col], errors="coerce"); data[value_col] = pd.to_numeric(data[value_col], errors="coerce"); data = data.dropna().sort_values(date_col).set_index(date_col)
                if len(data) >= 2:
                    st.line_chart(data[value_col]); window = st.slider("Rolling window", 2, max(2, min(100, len(data))), min(7, max(2, len(data)))); mean = data[value_col].rolling(window).mean(); sd = data[value_col].rolling(window).std(); st.subheader("Rolling mean"); st.line_chart(mean); st.subheader("Rolling standard deviation"); st.line_chart(sd); download_csv(pd.DataFrame({"Value": data[value_col], "Rolling mean": mean, "Rolling standard deviation": sd}).reset_index(), "time_series.csv")
                else: st.warning("Not enough valid date/value observations.")

elif page == "Bootstrap":
    st.header("Bootstrap resampling")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if nums:
            col = st.selectbox("Variable", nums); iterations = st.number_input("Iterations", 100, 10000, 1000, 100); confidence = st.selectbox("Confidence level", [90, 95, 99], index=1); statistic = st.selectbox("Statistic", ["Mean", "Median"])
            if st.button("Run bootstrap"):
                x = clean_numeric(df[col]).values
                if len(x):
                    vals = [np.mean(np.random.choice(x, size=len(x), replace=True)) if statistic == "Mean" else np.median(np.random.choice(x, size=len(x), replace=True)) for _ in range(int(iterations))]; vals = np.asarray(vals); alpha = 1 - confidence / 100; lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]); st.metric(f"Bootstrap {statistic}", float(np.mean(vals))); st.metric("Lower bound", float(lo)); st.metric("Upper bound", float(hi)); res = pd.DataFrame({"Bootstrap statistic": vals}); st.dataframe(res.head(100)); download_csv(res, "bootstrap.csv")

elif page == "Effect size":
    st.header("Effect size (Cohen's d)")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums, cats = numeric_columns(df), categorical_columns(df)
        if nums and cats:
            col = st.selectbox("Numeric variable", nums); group_col = st.selectbox("Group variable", cats); levels = df[group_col].dropna().unique().tolist()
            if len(levels) >= 2:
                g1 = st.selectbox("Group 1", levels); g2 = st.selectbox("Group 2", [v for v in levels if v != g1])
                if st.button("Calculate"):
                    x = clean_numeric(df.loc[df[group_col] == g1, col]); y = clean_numeric(df.loc[df[group_col] == g2, col])
                    if len(x) > 1 and len(y) > 1:
                        pooled = np.sqrt(((len(x)-1)*x.var() + (len(y)-1)*y.var()) / (len(x)+len(y)-2))
                        if pooled > 0: st.metric("Cohen's d", float((x.mean()-y.mean())/pooled))
                        else: st.warning("Pooled standard deviation is zero.")

elif page == "Power analysis":
    st.header("Power analysis context")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        nums = numeric_columns(df)
        if nums:
            col = st.selectbox("Variable", nums); mu = st.number_input("Reference mean", value=0.0); x = clean_numeric(df[col]); st.metric("Sample size", len(x))
            if len(x) > 1:
                st.metric("Mean", float(x.mean())); st.metric("Standard deviation", float(x.std())); st.metric("Observed standardized effect", float((x.mean()-mu)/x.std()) if x.std() > 0 else 0.0)
                st.info("Observed effect is descriptive. Post-hoc power based on observed effects does not replace prospective power analysis.")

elif page == "Profiling":
    st.header("Data profiling")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        rows = []
        for col in df.columns:
            s = df[col]; rows.append({"Column": col, "Type": str(s.dtype), "Rows": len(s), "Missing": int(s.isna().sum()), "Missing (%)": s.isna().sum()/max(len(s),1)*100, "Unique": int(s.nunique()), "Memory bytes": int(s.memory_usage(deep=True))})
        profile = pd.DataFrame(rows); st.dataframe(profile, use_container_width=True); st.write({"Rows": len(df), "Columns": len(df.columns), "Duplicates": int(df.duplicated().sum()), "Missing cells": int(df.isna().sum().sum()), "Numeric": len(numeric_columns(df)), "Categorical": len(categorical_columns(df))}); download_csv(profile, "profile.csv")

elif page == "Reports":
    st.header("Dataset report")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        report = {"project": "BioDataLab", "generated_at": datetime.now().isoformat(), "rows": len(df), "columns": len(df.columns), "column_names": list(df.columns), "types": {c: str(t) for c,t in df.dtypes.items()}, "missing": {c: int(v) for c,v in df.isna().sum().items()}, "duplicates": int(df.duplicated().sum()), "statistics": df.describe().to_dict(), "history": st.session_state.history}; st.write({"Rows": len(df), "Columns": len(df.columns), "Missing": int(df.isna().sum().sum()), "Duplicates": int(df.duplicated().sum())}); st.dataframe(df.describe(), use_container_width=True); st.subheader("Analysis history"); st.dataframe(pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()); download_json(report, "biodatalab_report.json")

elif page == "Export":
    st.header("Export data")
    df = st.session_state.data
    if df is None: st.warning("Upload a dataset first.")
    else:
        st.dataframe(df, use_container_width=True); download_csv(df, "biodatalab_dataset.csv"); buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Dataset", index=False)
            if numeric_columns(df): df.describe().to_excel(writer, sheet_name="Statistics")
        st.download_button("Download Excel workbook", buffer.getvalue(), "biodatalab.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.session_state.results: download_json(st.session_state.results, "analysis_results.json")

st.divider()
st.caption("BioDataLab | Open-source scientific data analysis")
