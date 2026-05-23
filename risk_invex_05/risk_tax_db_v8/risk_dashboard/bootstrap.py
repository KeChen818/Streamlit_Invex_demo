"""Shared Streamlit bootstrapping helpers for dashboard pages."""

import pandas as pd
import streamlit as st

from .data import filter_data, load_json_records, normalize_data


def load_filtered_inventory() -> pd.DataFrame:
    """Load the bundled inventory and apply the shared sidebar filters."""
    try:
        records = load_json_records()
    except Exception as exc:
        st.error(f"Unable to load JSON: {exc}")
        st.stop()

    if not records:
        st.error("No risk records found in the selected JSON.")
        st.stop()

    data = normalize_data(records)
    with st.sidebar:
        st.header("Filters")
        filtered = filter_data(data)

    if filtered.empty:
        st.warning("No records match the current filters.")
        st.stop()

    return filtered
