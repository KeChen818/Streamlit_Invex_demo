"""Streamlit page configuration and CSS for the dashboard."""

import streamlit as st

from .settings import APP_TITLE


def configure_page(page_title: str | None = None) -> None:
    """Configure Streamlit's page metadata and layout."""
    st.set_page_config(
        page_title=page_title or APP_TITLE,
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
        .risk-table-wrap {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff;
            max-height: 460px;
            overflow: auto;
            width: 100%;
        }
        .risk-table-wrap-wide {
            overflow-x: auto;
            overflow-y: auto;
        }
        .risk-html-table {
            background: #fff;
            border-collapse: separate;
            border-spacing: 0;
            color: var(--ink);
            font-size: 0.82rem;
            line-height: 1.35;
            width: 100%;
        }
        .risk-table-wrap-wide .risk-html-table {
            min-width: 1680px;
            width: max-content;
        }
        .risk-html-table thead th {
            background: var(--line-soft);
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            font-weight: 700;
            padding: 0.56rem 0.68rem;
            position: sticky;
            text-align: left;
            top: 0;
            z-index: 1;
        }
        .risk-html-table tbody tr,
        .risk-html-table tbody td {
            background: #fff !important;
        }
        .risk-html-table tbody td {
            border-bottom: 1px solid var(--line-soft);
            color: var(--ink);
            padding: 0.52rem 0.68rem;
            vertical-align: top;
        }
        .risk-html-table .col-taxonomy-l1 {
            font-weight: 700;
            min-width: 150px;
        }
        .risk-html-table .col-risks {
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            font-weight: 700;
            text-align: right;
            white-space: nowrap;
        }
        .risk-html-table .col-associated-sublegal-entity,
        .risk-html-table .col-associated-business-division,
        .risk-html-table .col-metrics,
        .risk-html-table .col-methods {
            max-width: 260px;
            min-width: 180px;
            white-space: normal;
        }
        .risk-table-wrap-wide .risk-html-table th,
        .risk-table-wrap-wide .risk-html-table td {
            min-width: 130px;
            white-space: nowrap;
        }
        .risk-table-wrap-wide .risk-html-table th:nth-child(2),
        .risk-table-wrap-wide .risk-html-table td:nth-child(2),
        .risk-table-wrap-wide .risk-html-table th:nth-child(4),
        .risk-table-wrap-wide .risk-html-table td:nth-child(4),
        .risk-table-wrap-wide .risk-html-table th:nth-child(5),
        .risk-table-wrap-wide .risk-html-table td:nth-child(5),
        .risk-table-wrap-wide .risk-html-table th:nth-child(8),
        .risk-table-wrap-wide .risk-html-table td:nth-child(8),
        .risk-table-wrap-wide .risk-html-table th:nth-child(10),
        .risk-table-wrap-wide .risk-html-table td:nth-child(10),
        .risk-table-wrap-wide .risk-html-table th:nth-child(13),
        .risk-table-wrap-wide .risk-html-table td:nth-child(13),
        .risk-table-wrap-wide .risk-html-table th:nth-child(14),
        .risk-table-wrap-wide .risk-html-table td:nth-child(14) {
            min-width: 240px;
            white-space: normal;
        }
        .risk-html-table tbody tr:hover td {
            background: #fff !important;
        }
        .score-bar-cell {
            align-items: center;
            display: flex;
            gap: 0.5rem;
            min-width: 116px;
        }
        .score-bar-track {
            background: var(--line-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            height: 7px;
            min-width: 72px;
            overflow: hidden;
        }
        .score-bar-fill {
            background: var(--accent);
            display: block;
            height: 100%;
        }
        .score-bar-label {
            color: var(--ink-soft);
            font-size: 0.74rem;
            font-variant-numeric: tabular-nums;
            min-width: 2rem;
        }
        .bool-badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            line-height: 1;
            min-width: 42px;
            padding: 0.28rem 0.45rem;
            text-align: center;
            text-transform: uppercase;
        }
        .bool-badge-true {
            background: #f8d7d7;
            color: #9f2f2f;
        }
        .bool-badge-false {
            background: var(--line-soft);
            color: var(--ink-soft);
        }
        .review-controls-title {
            color: var(--ink-soft);
            font-size: 0.82rem;
            font-weight: 700;
            margin: 0.9rem 0 0.35rem;
            text-transform: uppercase;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #fff !important;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stDataEditor"] > div {
            background: var(--bg-card) !important;
        }
        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataEditor"] canvas,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataEditor"] [role="grid"],
        div[data-testid="stDataFrame"] [role="rowgroup"],
        div[data-testid="stDataEditor"] [role="rowgroup"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataEditor"] [role="row"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataEditor"] [role="gridcell"] {
            background: #fff !important;
        }
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataEditor"] [role="columnheader"],
        div[data-testid="stDataFrame"] thead th,
        div[data-testid="stDataEditor"] thead th {
            background: var(--line-soft) !important;
            color: var(--ink) !important;
        }
        div[data-testid="stDataFrame"] tbody tr,
        div[data-testid="stDataFrame"] tbody td,
        div[data-testid="stDataEditor"] tbody tr,
        div[data-testid="stDataEditor"] tbody td {
            background: #fff !important;
            color: var(--ink) !important;
        }
        div[data-testid="stExpander"] {
            background: var(--bg-card);
            border-color: var(--line) !important;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] details,
        details[data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border-color: var(--line) !important;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary,
        details[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] details > summary {
            background: var(--line-soft) !important;
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary:hover,
        details[data-testid="stExpander"] summary:hover,
        div[data-testid="stExpander"] details:hover > summary {
            background: var(--line) !important;
        }
        div[data-testid="stExpander"] summary *,
        details[data-testid="stExpander"] summary * {
            background: transparent !important;
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
