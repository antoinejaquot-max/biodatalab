











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




