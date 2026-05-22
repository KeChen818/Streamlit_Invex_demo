"""Small formatting and parsing helpers used across the dashboard."""

import re

import pandas as pd


def unique_join(series: pd.Series, limit: int = 4) -> str:
    """Join unique values into a compact label for table and card summaries."""
    values = sorted({str(value).strip() for value in series.dropna() if str(value).strip()})
    if not values:
        return "none"
    if len(values) > limit:
        return ", ".join(values[:limit]) + f" +{len(values) - limit}"
    return ", ".join(values)


def top_counts(series: pd.Series, limit: int = 2) -> list[tuple[str, int]]:
    """Return top value counts as label/count tuples."""
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    counts = clean.value_counts().head(limit)
    return [(str(label), int(count)) for label, count in counts.items()]


def format_top_counts(counts: list[tuple[str, int]]) -> str:
    """Format top-count tuples for narrative summaries."""
    if not counts:
        return "none"
    return ", ".join(f"{label} ({count})" for label, count in counts)


def format_mapping(mapping: dict[str, str], limit: int = 4) -> str:
    """Format a short key/value mapping for AI fallback text."""
    items = [f"{key}: {value}" for key, value in mapping.items()]
    if len(items) > limit:
        return "; ".join(items[:limit]) + f"; +{len(items) - limit} more"
    return "; ".join(items)


def extract_assumption_basis(comment: str) -> str:
    """Extract the assumption basis phrase from an impact commentary sentence."""
    clean = re.sub(r"\s+", " ", str(comment or "")).strip()
    if not clean:
        return "no commentary provided"

    match = re.search(r"applied to (.*?) with conservative", clean, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return clean[:180] + ("..." if len(clean) > 180 else "")
