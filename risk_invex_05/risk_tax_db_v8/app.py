"""Streamlit entrypoint and page navigation for the risk taxonomy dashboard."""

from pathlib import Path

import streamlit as st

from risk_dashboard.main import main as render_taxonomy_grouping
from risk_dashboard.settings import APP_TITLE


AI_THEME_PAGE = Path(__file__).parent / "pages" / "2_AI_Risk_Theme_Classification.py"


def main() -> None:
    """Route between the dashboard pages with user-facing sidebar labels."""
    selected_page = st.navigation(
        [
            st.Page(
                render_taxonomy_grouping,
                title=APP_TITLE,
                icon=":material/account_tree:",
                url_path="",
                default=True,
            ),
            st.Page(
                AI_THEME_PAGE,
                title="AI Risk Theme Classification",
                icon=":material/hub:",
                url_path="AI_Risk_Theme_Classification",
            ),
        ]
    )
    selected_page.run()


if __name__ == "__main__":
    main()
