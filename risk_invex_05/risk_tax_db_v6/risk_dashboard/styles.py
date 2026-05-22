"""Streamlit page configuration and CSS for the dashboard."""

import streamlit as st

from .settings import APP_TITLE


def configure_page() -> None:
    """Configure Streamlit's page metadata and layout."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="RT",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    """Apply the dashboard theme and custom card styling."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5f2ec;
            --bg-card: #fff;
            --ink: #1a1814;
            --ink-soft: #5c5650;
            --line: #e3ddd2;
            --line-soft: #ecebe6;
            --accent: #d64545;
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        header[data-testid="stHeader"],
        div[data-testid="stDecoration"] {
            background: var(--bg);
        }
        section[data-testid="stSidebar"] {
            background: #eee8de;
            border-right: 1px solid var(--line);
        }
        h1, h2, h3, p, label {
            letter-spacing: 0 !important;
            color: var(--ink);
        }
        h1 {
            font-size: 1.55rem !important;
            margin-bottom: 0.1rem !important;
        }
        h2 {
            font-size: 1.05rem !important;
            margin-top: 0.5rem !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: var(--bg-card);
            min-height: 98px;
            box-shadow: 0 1px 1px rgba(26, 24, 20, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: var(--ink-soft);
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.45rem;
        }
        div[data-testid="stMetricDelta"] {
            color: var(--accent);
        }
        div[data-testid="stTabs"] button {
            border-radius: 8px 8px 0 0;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom-color: var(--accent);
            color: var(--accent);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--bg-card);
        }
        div[data-testid="stExpander"] {
            background: var(--bg-card);
            border-color: var(--line) !important;
            border-radius: 8px;
        }
        .taxonomy-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--bg-card);
            min-height: 148px;
            padding: 0.9rem;
            margin-bottom: 0.55rem;
            box-shadow: 0 1px 1px rgba(26, 24, 20, 0.04);
        }
        .taxonomy-card h3 {
            font-size: 1rem;
            line-height: 1.25;
            margin: 0 0 0.6rem;
        }
        .taxonomy-meta {
            color: var(--ink-soft);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-bottom: 0.35rem;
        }
        .risk-pop-card {
            border: 1px solid var(--line-soft);
            border-radius: 8px;
            background: #fffaf2;
            padding: 0.75rem;
            margin: 0 0 0.6rem;
        }
        .risk-pop-card b {
            color: var(--ink);
            display: block;
            font-size: 0.92rem;
            line-height: 1.3;
            margin-bottom: 0.35rem;
        }
        .risk-pop-card span {
            color: var(--ink-soft);
            display: block;
            font-size: 0.78rem;
            line-height: 1.45;
        }
        .stButton button, .stDownloadButton button {
            border-radius: 8px;
            border-color: var(--line);
            color: var(--ink);
            background: var(--bg-card);
        }
        .stDownloadButton button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }
        .ai-card-title {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0 !important;
            margin-bottom: 0.4rem;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
