"""Data loading, normalization, and filtering helpers."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .settings import DATA_PATH, EXPECTED_COLUMNS, REQUIRED_COLUMNS


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
    # Materiality is a direct inventory flag from Overall_Materiality, not an AI-derived classification.
    df["Is_Material"] = df["Overall_Materiality"].str.lower().eq("material")
    return df


def sorted_options(data: pd.DataFrame, column: str) -> list[str]:
    """Build stable non-empty filter choices for a dataframe column."""
    return sorted(value for value in data[column].dropna().unique().tolist() if str(value).strip())


def risk_lookup_options(data: pd.DataFrame) -> dict[str, str]:
    """Build searchable Group ID labels that include each risk title."""
    records = data[["Group_ID", "Risk_Title"]].drop_duplicates().sort_values(["Group_ID", "Risk_Title"])
    return {
        str(row.Group_ID): f"{row.Group_ID} | {row.Risk_Title}"
        for row in records.itertuples(index=False)
        if str(row.Group_ID).strip()
    }


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar slicers and selected-risk lookup to the inventory."""
    filtered = data.copy()

    filter_map = {
        "SubLegal_Entity": "SubLegal_Entity",
        "Risk Type": "Risk_Type",
        "Taxonomy L0": "Taxonomy_L0",
        "Business Division": "Business_Division",
        "GCRS": "GCRS",
        "Overall Materiality": "Overall_Materiality",
    }

    for label, column in filter_map.items():
        options = sorted_options(data, column)
        default = ["Financial"] if column == "Risk_Type" and "Financial" in options else []
        choices = st.sidebar.multiselect(label, options, default=default)
        if choices:
            filtered = filtered[filtered[column].isin(choices)]

    lookup_labels = risk_lookup_options(filtered)
    selected_group_ids = st.sidebar.multiselect(
        "Group ID / Risk Title",
        list(lookup_labels.keys()),
        format_func=lambda group_id: lookup_labels.get(group_id, group_id),
    )
    if selected_group_ids:
        filtered = filtered[filtered["Group_ID"].astype(str).isin(selected_group_ids)]

    return filtered
