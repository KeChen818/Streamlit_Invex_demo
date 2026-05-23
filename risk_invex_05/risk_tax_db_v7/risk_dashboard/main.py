"""Application orchestration for the Streamlit dashboard."""

import streamlit as st

from .ai import selected_model
from .analysis import build_group_summary, build_method_metric
from .data import filter_data, load_json_records, normalize_data
from .settings import APP_TITLE
from .styles import configure_page, inject_css
from .views import (
    metric_cards,
    render_ai_chatbot,
    render_comparison,
    render_group_summary,
    render_theme_classification,
)


def main() -> None:
    """Run the Streamlit dashboard."""
    configure_page()
    inject_css()
    model = selected_model()

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

    st.title(APP_TITLE)

    if filtered.empty:
        st.warning("No records match the current filters.")
        st.stop()

    group_summary = build_group_summary(filtered)
    comparison = build_method_metric(filtered)

    metric_cards(filtered)

    tab_summary, tab_compare, tab_theme, tab_chat = st.tabs(
        ["Taxonomy L1 Groups", "Method and Metric Compare", "AI Theme Classification", "AI Chatbot"],
    )

    with tab_summary:
        render_group_summary(filtered, group_summary, model)

    with tab_compare:
        render_comparison(comparison, group_summary, filtered, model)

    with tab_theme:
        render_theme_classification(filtered, model)

    with tab_chat:
        render_ai_chatbot(filtered, model)

    st.sidebar.divider()
    st.sidebar.download_button(
        "Export group summary",
        data=group_summary.to_csv(index=False).encode("utf-8"),
        file_name="taxonomy_l1_group_summary.csv",
        mime="text/csv",
    )
