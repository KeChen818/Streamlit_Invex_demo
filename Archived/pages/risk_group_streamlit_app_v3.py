
"""
Streamlit MVP: Risk Group Generator with compact IDs

Run:
  pip install streamlit pandas
  streamlit run risk_group_streamlit_app_v3.py

Input accepted:
  - JSON array: [{...}, {...}]
  - Single JSON object: {...}
  - CSV with matching column names

Compact grouping rules:
  1. local_group_id  = RG-L-[6-char hash]
     Keys: sub_legal_entity + business_division + risk_type + taxonomy_l0 + taxonomy_l1 + taxonomy_l2
  2. theme_group_id  = RG-T-[6-char hash]
     Keys: risk_type + taxonomy_l0 + taxonomy_l1
  3. entity_group_id = RG-E-[6-char hash]
     Keys: sub_legal_entity + risk_type + taxonomy_l0 + taxonomy_l1

Readable labels are generated separately as local_group_label, theme_group_label, and entity_group_label.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

GROUP_RULES: Dict[str, Dict[str, object]] = {
    "Local Business Group": {
        "id_column": "local_group_id",
        "label_column": "local_group_label",
        "prefix": "RG-L",
        "keys": ["sub_legal_entity", "business_division", "risk_type", "taxonomy_l0", "taxonomy_l1", "taxonomy_l2"],
        "description": "Groups risks within the same sub-legal entity and business division when they share the same detailed taxonomy path.",
    },
    "Cross-Business Theme Group": {
        "id_column": "theme_group_id",
        "label_column": "theme_group_label",
        "prefix": "RG-T",
        "keys": ["risk_type", "taxonomy_l0", "taxonomy_l1"],
        "description": "Groups risks across business divisions when they share the same core taxonomy theme.",
    },
    "Entity Group": {
        "id_column": "entity_group_id",
        "label_column": "entity_group_label",
        "prefix": "RG-E",
        "keys": ["sub_legal_entity", "risk_type", "taxonomy_l0", "taxonomy_l1"],
        "description": "Groups risks within the same sub-legal entity when they share the same core taxonomy theme.",
    },
}

REQUIRED_COLUMNS = sorted({col for rule in GROUP_RULES.values() for col in rule["keys"]})
MATERIALITY_RANK = {"non-material": 0, "non material": 0, "low": 1, "medium": 2, "high": 3, "material": 4}
RATING_COLUMNS = ["likelihood_rating", "impact_rating"]


def normalize_text(value: object) -> str:
    if pd.isna(value) or value is None:
        return "unknown"
    text = str(value).strip().lower().replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text or "unknown"


def stable_hash(values: Iterable[object], length: int = 6) -> str:
    key = "|".join(normalize_text(v) for v in values)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:length].upper()


def clean_label(value: object) -> str:
    text = str(value).strip() if value is not None and not pd.isna(value) else "Unknown"
    return re.sub(r"\s+", " ", text)


def generate_compact_group_id(row: pd.Series, prefix: str, keys: List[str]) -> str:
    return f"{prefix}-{stable_hash([row.get(k) for k in keys])}"


def generate_group_label(row: pd.Series, keys: List[str]) -> str:
    return " / ".join(clean_label(row.get(k)) for k in keys)


def join_unique(values: pd.Series, limit: int = 8) -> str:
    unique: List[str] = []
    for value in values.dropna().astype(str):
        for item in re.split(r";|,\s(?=[\w.-]+@[\w.-]+)|\n", value):
            item = item.strip()
            if item and item.lower() not in {"nan", "none", "unknown"} and item not in unique:
                unique.append(item)
    suffix = " ..." if len(unique) > limit else ""
    return "; ".join(unique[:limit]) + suffix


def materiality_max(values: pd.Series) -> str:
    labels = [str(v).strip() for v in values.dropna().tolist() if str(v).strip()]
    if not labels:
        return "Unknown"
    return max(labels, key=lambda x: MATERIALITY_RANK.get(x.lower(), -1))


def range_text(values: pd.Series) -> str:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return "Unknown"
    if nums.min() == nums.max():
        return f"{nums.min():g}"
    return f"{nums.min():g} - {nums.max():g}"


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raw = uploaded_file.read().decode("utf-8")
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def load_demo_data() -> pd.DataFrame:
    sample_path = Path(__file__).with_name("demo_risk_inventory_24_records.json")
    if sample_path.exists():
        return pd.read_json(sample_path)
    return pd.DataFrame([])


def add_all_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required grouping columns: {missing}")
    out = df.copy()
    for rule in GROUP_RULES.values():
        id_col = str(rule["id_column"])
        label_col = str(rule["label_column"])
        prefix = str(rule["prefix"])
        keys = list(rule["keys"])
        out[id_col] = out.apply(lambda row: generate_compact_group_id(row, prefix, keys), axis=1)
        out[label_col] = out.apply(lambda row: generate_group_label(row, keys), axis=1)
    return out


def build_group_explanation(row: pd.Series, selected_view: str) -> str:
    risk_path_l2 = f"{row.get('risk_type', 'Unknown')} > {row.get('taxonomy_l0', 'Unknown')} > {row.get('taxonomy_l1', 'Unknown')} > {row.get('taxonomy_l2', 'Unknown')}"
    risk_path_l1 = f"{row.get('risk_type', 'Unknown')} > {row.get('taxonomy_l0', 'Unknown')} > {row.get('taxonomy_l1', 'Unknown')}"
    if selected_view == "Local Business Group":
        return f"This local group includes risks within {row.get('sub_legal_entity', 'Unknown')} / {row.get('business_divisions', 'Unknown')} that share the detailed taxonomy path {risk_path_l2}. These risks are likely related from a business activity, control, ownership, and assessment perspective."
    if selected_view == "Cross-Business Theme Group":
        return f"This theme group includes risks across business divisions that share the same taxonomy theme {risk_path_l1}. The purpose is to support horizontal comparison, taxonomy consistency review, duplicate risk detection, and review of rating differences across businesses."
    return f"This entity group includes risks within {row.get('sub_legal_entity', 'Unknown')} that share the taxonomy theme {risk_path_l1}. The purpose is to support legal-entity-level aggregation, committee reporting, and consistency review across business divisions within the same entity."


def build_group_suggestions(group: pd.DataFrame, selected_view: str) -> str:
    suggestions: List[str] = []
    if len(group) > 1:
        suggestions.append("Review whether grouped risk narratives are duplicates, overlaps, or intentionally separate records.")
    if "business_division" in group.columns and group["business_division"].nunique(dropna=True) > 1:
        suggestions.append("Compare rating rationale across business divisions because the same taxonomy theme appears in multiple businesses.")
    if "sub_legal_entity" in group.columns and group["sub_legal_entity"].nunique(dropna=True) > 1:
        suggestions.append("Confirm legal-entity consistency because the same theme appears across multiple sub-legal entities.")
    if "materiality" in group.columns and group["materiality"].nunique(dropna=True) > 1:
        suggestions.append("Review materiality calibration because the group contains mixed materiality outcomes.")
    for col in RATING_COLUMNS:
        if col in group.columns:
            nums = pd.to_numeric(group[col], errors="coerce").dropna()
            if not nums.empty and nums.max() - nums.min() >= 2:
                suggestions.append(f"Review {col.replace('_', ' ')} consistency because ratings vary materially within the group.")
    if "lod1_emails" in group.columns and group["lod1_emails"].nunique(dropna=True) > 1:
        suggestions.append("Confirm ownership because multiple 1LoD owners appear in the same group.")
    if "root_cause_l1" in group.columns and group["root_cause_l1"].nunique(dropna=True) > 1:
        suggestions.append("Check whether different root causes should remain separate or be summarized under one broader scenario driver.")
    if not suggestions:
        suggestions.append("Use this group for the selected review lens and document any SME rationale for keeping records separate.")
    return " ".join(f"{i + 1}) {s}" for i, s in enumerate(suggestions[:5]))


def build_grouped_view(df_with_ids: pd.DataFrame, selected_view: str) -> Tuple[pd.DataFrame, str, str]:
    rule = GROUP_RULES[selected_view]
    group_col = str(rule["id_column"])
    label_col = str(rule["label_column"])
    keys = list(rule["keys"])
    grouped_rows: List[Dict[str, object]] = []
    for group_id, group in df_with_ids.groupby(group_col, dropna=False):
        first = group.iloc[0]
        row: Dict[str, object] = {
            "selected_group_type": selected_view,
            "group_id": group_id,
            "group_label": first.get(label_col, "Unknown"),
            "risk_count": len(group),
            "sub_legal_entity": first.get("sub_legal_entity", "Unknown"),
            "business_divisions": join_unique(group.get("business_division", pd.Series(dtype=str)), limit=10),
            "risk_type": first.get("risk_type", "Unknown"),
            "taxonomy_l0": first.get("taxonomy_l0", "Unknown"),
            "taxonomy_l1": first.get("taxonomy_l1", "Unknown"),
            "taxonomy_l2": first.get("taxonomy_l2", "Unknown") if "taxonomy_l2" in keys else "Multiple / Not used",
            "risk_titles": join_unique(group.get("risk_title", pd.Series(dtype=str)), limit=10),
            "risk_status_mix": join_unique(group.get("risk_status", pd.Series(dtype=str)), limit=6),
            "max_materiality": materiality_max(group.get("materiality", pd.Series(dtype=str))),
            "likelihood_range": range_text(group.get("likelihood_rating", pd.Series(dtype=float))),
            "impact_rating_range": range_text(group.get("impact_rating", pd.Series(dtype=float))),
            "max_impact_amount_usd": pd.to_numeric(group.get("impact_amount_usd", pd.Series(dtype=float)), errors="coerce").max(),
            "lod1_emails": join_unique(group.get("lod1_emails", pd.Series(dtype=str)), limit=12),
            "lod2_emails": join_unique(group.get("lod2_emails", pd.Series(dtype=str)), limit=12),
            "root_causes": join_unique(group.get("root_cause_l1", pd.Series(dtype=str)), limit=8),
            "source_record_ids": join_unique(group.get("group_id", pd.Series(dtype=str)), limit=12),
        }
        row["group_explanation"] = build_group_explanation(pd.Series(row), selected_view)
        row["group_suggestions"] = build_group_suggestions(group, selected_view)
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)
    if not grouped.empty:
        grouped = grouped.sort_values(["risk_count", "group_id"], ascending=[False, True])
    return grouped, group_col, label_col


st.set_page_config(page_title="Risk Group Generator", layout="wide")
st.title("Risk Group Generator")
st.caption("Compact group IDs + readable group labels + explanations + review suggestions.")

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload risk inventory JSON or CSV", type=["json", "csv"])
    use_demo = st.toggle("Use 24-record demo data", value=uploaded_file is None)

    st.header("Grouping View")
    selected_view = st.radio("Choose grouping lens", options=list(GROUP_RULES.keys()), index=0)
    st.info(str(GROUP_RULES[selected_view]["description"]))

    with st.expander("Compact ID design", expanded=False):
        st.markdown("IDs are intentionally short and stable:")
        st.code("RG-L-[hash]  Local Business Group\nRG-T-[hash]  Cross-Business Theme Group\nRG-E-[hash]  Entity Group", language="text")
        st.markdown("Readable details are stored separately in the group label columns.")

try:
    if uploaded_file is not None and not use_demo:
        raw_df = load_uploaded_file(uploaded_file)
    else:
        raw_df = load_demo_data()

    df_with_ids = add_all_group_ids(raw_df)
    grouped_df, active_group_col, active_label_col = build_grouped_view(df_with_ids, selected_view)

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Raw risk records", len(raw_df))
    top2.metric("Selected groups", len(grouped_df))
    top3.metric("Grouped records", max(len(raw_df) - len(grouped_df), 0))
    top4.metric("Active ID", active_group_col)

    st.subheader("Grouping Rules")
    st.markdown("""
- **Local Business Group:** `sub_legal_entity + business_division + risk_type + taxonomy_l0 + taxonomy_l1 + taxonomy_l2`
- **Cross-Business Theme Group:** `risk_type + taxonomy_l0 + taxonomy_l1`
- **Entity Group:** `sub_legal_entity + risk_type + taxonomy_l0 + taxonomy_l1`

**Simplified ID format:** `RG-L-ABC123`, `RG-T-ABC123`, or `RG-E-ABC123`. The readable business context is stored in `group_label`.
""")

    tab_grouped, tab_raw, tab_ids = st.tabs(["Grouped View", "Raw Records", "Generated IDs"])

    with tab_grouped:
        st.subheader(f"Generated Groups: {selected_view}")
        display_cols = ["group_id", "group_label", "risk_count", "sub_legal_entity", "business_divisions", "risk_type", "taxonomy_l0", "taxonomy_l1", "taxonomy_l2", "max_materiality", "likelihood_range", "impact_rating_range", "max_impact_amount_usd", "risk_titles", "group_explanation", "group_suggestions", "lod1_emails", "lod2_emails", "root_causes", "source_record_ids"]
        st.dataframe(grouped_df[[c for c in display_cols if c in grouped_df.columns]], use_container_width=True, height=560)
        st.download_button("Download selected grouped view as CSV", grouped_df.to_csv(index=False).encode("utf-8"), file_name=f"generated_{active_group_col}.csv", mime="text/csv")

    with tab_raw:
        st.subheader("Raw Risk Records")
        st.dataframe(raw_df, use_container_width=True, height=560)

    with tab_ids:
        st.subheader("Raw Records with Compact IDs and Readable Labels")
        id_cols = ["local_group_id", "local_group_label", "theme_group_id", "theme_group_label", "entity_group_id", "entity_group_label"]
        preview_cols = [c for c in ["group_id", "risk_title", "sub_legal_entity", "business_division", "risk_type", "taxonomy_l0", "taxonomy_l1", "taxonomy_l2", *id_cols] if c in df_with_ids.columns]
        st.dataframe(df_with_ids[preview_cols], use_container_width=True, height=560)
        st.download_button("Download raw records with all group IDs as CSV", df_with_ids.to_csv(index=False).encode("utf-8"), file_name="risk_records_with_compact_group_ids.csv", mime="text/csv")

except Exception as exc:
    st.error(str(exc))
    st.info("Check that your file includes the required columns: " + ", ".join(REQUIRED_COLUMNS) + ".")
