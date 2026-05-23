"""Data loading, normalization, and filtering helpers."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .settings import DATA_PATH, EXPECTED_COLUMNS, REQUIRED_COLUMNS, SEARCH_COLUMNS


def records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Return a list of risk dictionaries from common JSON payload shapes."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []
        for key in ("records", "risks", "data", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if not records:
            records = [payload]
    else:
        records = []

    return [record for record in records if isinstance(record, dict)]


def load_json_records(data_path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Load the private bundled inventory JSON used by the app."""
    with data_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return records_from_payload(payload)


def normalize_data(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate required fields and standardize blanks before analysis."""
    df = pd.DataFrame(records)

    missing_required = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_required:
        st.error("Missing required columns: " + ", ".join(missing_required))
        st.stop()

    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[EXPECTED_COLUMNS + [col for col in df.columns if col not in EXPECTED_COLUMNS]].copy()

    text_columns = [col for col in EXPECTED_COLUMNS if col != "Impact_Numbers"]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["Group_ID"] = df["Group_ID"].where(
        df["Group_ID"].ne(""),
        [f"RISK-{idx + 1:03d}" for idx in range(len(df))],
    )
    df["Taxonomy_L1"] = df["Taxonomy_L1"].where(df["Taxonomy_L1"].ne(""), "Unmapped")
    df["Risk_Metric"] = df["Risk_Metric"].where(df["Risk_Metric"].ne(""), "Unspecified")
    df["Assessment_Method"] = df["Assessment_Method"].where(
        df["Assessment_Method"].ne(""),
        "Unspecified",
    )
    df["Likelihood_Score"] = (
        df["Likelihood_Rating"].str.extract(r"^\s*(\d+)")[0].astype(float).fillna(0)
    )
    df["Is_Material"] = df["Overall_Materiality"].str.lower().eq("material")
    return df


def sorted_options(data: pd.DataFrame, column: str) -> list[str]:
    """Build stable non-empty filter choices for a dataframe column."""
    return sorted(value for value in data[column].dropna().unique().tolist() if str(value).strip())


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar slicers and keyword search to the inventory."""
    filtered = data.copy()

    filter_map = {
        "Reporting quarter": "Reporting_Quarter",
        "Taxonomy L0": "Taxonomy_L0",
        "Taxonomy L1 group": "Taxonomy_L1",
        "Business division": "Business_Division",
        "Risk type": "Risk_Type",
        "Materiality": "Overall_Materiality",
        "Status": "Risk_Status",
    }

    for label, column in filter_map.items():
        choices = st.sidebar.multiselect(label, sorted_options(data, column))
        if choices:
            filtered = filtered[filtered[column].isin(choices)]

    search = st.sidebar.text_input("Search")
    clean_search = search.strip().lower()
    if clean_search:
        available_search_columns = [col for col in SEARCH_COLUMNS if col in filtered.columns]
        searchable = (
            filtered[available_search_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        filtered = filtered[searchable.str.contains(clean_search, regex=False, na=False)]

    return filtered
