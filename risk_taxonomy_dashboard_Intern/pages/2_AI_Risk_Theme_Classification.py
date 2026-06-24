"""Standalone Streamlit page for AI risk theme classification."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_dashboard.ai import selected_model
from risk_dashboard.bootstrap import load_filtered_inventory
from risk_dashboard.styles import configure_page, inject_css
from risk_dashboard.views import metric_cards, render_theme_classification


def main() -> None:
    """Render the dedicated AI Risk Theme Classification page."""
    configure_page("AI Risk Theme Classification")
    inject_css()
    model = selected_model()
    filtered = load_filtered_inventory()

    st.title("AI Risk Theme Classification")
    metric_cards(filtered)
    render_theme_classification(filtered, model, show_heading=False)


if __name__ == "__main__":
    main()
