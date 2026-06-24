"""Application orchestration for the Streamlit dashboard."""

import streamlit as st

from .ai import selected_model
from .analysis import build_group_summary, build_method_metric
from .bootstrap import load_filtered_inventory
from .settings import APP_TITLE
from .styles import configure_page, inject_css
from .views import (
    metric_cards,
    render_ai_chatbot,
    render_comparison,
    render_group_summary,
)


def main() -> None:
    """Run the Streamlit dashboard."""
    configure_page()
    inject_css()
    model = selected_model()
    filtered = load_filtered_inventory()

    st.title(APP_TITLE)

    group_summary = build_group_summary(filtered)
    comparison = build_method_metric(filtered)

    metric_cards(filtered)

    tab_summary, tab_compare, tab_chat = st.tabs(
        ["Taxonomy L1 Groups", "Method and Metric Compare", "AI Chatbot"],
    )

    with tab_summary:
        render_group_summary(filtered, group_summary, model)

    with tab_compare:
        render_comparison(comparison, group_summary, filtered, model)

    with tab_chat:
        render_ai_chatbot(filtered, model)

    st.sidebar.divider()
    st.sidebar.download_button(
        "Export group summary",
        data=group_summary.to_csv(index=False).encode("utf-8"),
        file_name="taxonomy_l1_group_summary.csv",
        mime="text/csv",
    )
