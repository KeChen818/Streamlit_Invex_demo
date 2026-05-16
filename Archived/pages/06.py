import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from services.context_builder import build_context
from services.skill_router import get_skill_instruction
from services.ai_client import ask_ai


# ======================================================
# Page config
# ======================================================

st.set_page_config(
    page_title="Inventory Chat — RISKEX",
    layout="wide",
)


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = APP_ROOT / "data" / "inventory.json"
SKILLS_DIR = APP_ROOT / "skills"
SKILLS_INDEX_PATH = SKILLS_DIR / "skills_index.json"


CUSTOM_CSS = """
<style>
.hero {
    padding: 1.1rem 1.3rem;
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.96), rgba(31, 41, 55, 0.92));
    border-radius: 16px;
    margin-bottom: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.25);
}
.hero-title {
    color: white;
    font-size: 1.55rem;
    font-weight: 700;
}
.hero-subtitle {
    color: #CBD5E1;
    font-size: 0.92rem;
    margin-top: 0.25rem;
}
.small-muted {
    color: #64748B;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <div class="hero-title">Inventory Chat</div>
  <div class="hero-subtitle">Schema-normalized inventory review, analytics, and AI-assisted risk profile Q&A</div>
</div>
""",
    unsafe_allow_html=True,
)


# ======================================================
# Schema normalization
# ======================================================

SCHEMA_MAP = {
    "group_id": ["Group_ID", "group_id"],
    "risk_name": ["Risk_Title", "risk_name", "Risk Name"],
    "risk_description": ["Risk_Description", "risk_description", "Risk Description"],
    "risk_status": ["Risk_Status", "status", "risk_status"],
    "business_division": ["Business_Division", "business_division", "Business Division"],
    "legal_entity": ["Legal_Entity", "legal_entity", "Legal Entity"],
    "risk_type": ["Risk_Type", "risk_category", "risk_type", "Risk Type"],
    "taxonomy_10": ["Taxonomy_L0", "taxonomy_10"],
    "taxonomy_11": ["Taxonomy_L1", "taxonomy_11"],
    "taxonomy_12": ["Taxonomy_L2", "taxonomy_12"],
    "overall_materiality": ["Overall_Materiality", "overall_materiality"],
    "assessment_method": ["Assessment_Method", "assessment_method"],
}


def normalize_inventory_schema(df: pd.DataFrame) -> pd.DataFrame:
    norm = {}

    for canon, candidates in SCHEMA_MAP.items():
        for c in candidates:
            if c in df.columns:
                norm[canon] = df[c]
                break
        else:
            norm[canon] = ""

    out = pd.DataFrame(norm)

    # Raw values if available
    out["likelihood"] = df.get("Likelihood_Rating", df.get("likelihood", ""))
    out["impact"] = df.get("Impact_Rating", df.get("impact", ""))

    # Optional QoQ fields
    optional_cols = [
        "previous_overall_materiality",
        "current_overall_materiality",
        "previous_likelihood",
        "current_likelihood",
        "previous_impact",
        "current_impact",
        "qoq_change",
        "change_rationale",
    ]
    for c in optional_cols:
        if c in df.columns:
            out[c] = df[c]

    return out


# ======================================================
# Data load
# ======================================================

@st.cache_data
def load_inventory(path: str) -> pd.DataFrame:
    p = Path(path)

    if not p.exists():
        return pd.DataFrame()

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


df_raw = load_inventory(str(DEFAULT_DATA_PATH))

if df_raw.empty:
    st.warning("No inventory file found. Please add data/inventory.json.")
    st.stop()

df = normalize_inventory_schema(df_raw)


# ======================================================
# Sidebar filters
# ======================================================

with st.sidebar:
    st.subheader("Filters")

    div_options = ["All"] + sorted(
        [x for x in df["business_division"].dropna().unique() if str(x).strip()]
    )
    risk_type_options = ["All"] + sorted(
        [x for x in df["risk_type"].dropna().unique() if str(x).strip()]
    )
    mat_options = ["All"] + sorted(
        [x for x in df["overall_materiality"].dropna().unique() if str(x).strip()]
    )

    div = st.selectbox("Business Division", div_options)
    rtype = st.selectbox("Risk Type", risk_type_options)
    materiality = st.selectbox("Materiality", mat_options)

    st.divider()

    st.caption("AI Settings")
    max_rows = st.slider("Max records sent to AI", 10, 80, 35, step=5)
    show_context = st.toggle("Show AI context preview", value=False)


# ======================================================
# Apply filtering
# ======================================================

filtered = df.copy()

if div != "All":
    filtered = filtered[filtered["business_division"] == div]

if rtype != "All":
    filtered = filtered[filtered["risk_type"] == rtype]

if materiality != "All":
    filtered = filtered[filtered["overall_materiality"] == materiality]


# ======================================================
# KPIs
# ======================================================

c1, c2, c3, c4 = st.columns(4)

material_count = (
    int((filtered["overall_materiality"] == "Material").sum())
    if "overall_materiality" in filtered.columns
    else 0
)

non_material_count = (
    int((filtered["overall_materiality"] == "Non-Material").sum())
    if "overall_materiality" in filtered.columns
    else 0
)

c1.metric("Total Risks", len(filtered))
c2.metric("Material", material_count)
c3.metric("Non-Material", non_material_count)
c4.metric("Risk Types", filtered["risk_type"].nunique())

st.divider()


# ======================================================
# Tabs
# ======================================================

tab_inv, tab_ana, tab_chat = st.tabs(
    [
        ":material/article: Inventory",
        ":material/analytics: Analytics",
        ":material/search: AI Chat",
    ]
)


# ======================================================
# Inventory tab
# ======================================================

with tab_inv:
    view = st.radio(
        "View Mode",
        ["Normalized", "Raw JSON"],
        horizontal=True,
    )

    if view == "Normalized":
        display_cols = [
            "group_id",
            "business_division",
            "legal_entity",
            "risk_type",
            "taxonomy_10",
            "taxonomy_11",
            "taxonomy_12",
            "risk_name",
            "likelihood",
            "impact",
            "overall_materiality",
            "risk_status",
            "assessment_method",
        ]

        display_cols = [c for c in display_cols if c in filtered.columns]

        st.dataframe(
            filtered[display_cols],
            use_container_width=True,
            height=520,
        )

        csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered CSV",
            csv,
            file_name="filtered_inventory.csv",
            mime="text/csv",
        )

    else:
        st.dataframe(df_raw, use_container_width=True, height=520)

        raw_csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Raw CSV",
            raw_csv,
            file_name="raw_inventory.csv",
            mime="text/csv",
        )


# ======================================================
# Analytics tab
# ======================================================

with tab_ana:
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Risk Type Distribution")
        if "risk_type" in filtered.columns:
            st.bar_chart(filtered["risk_type"].value_counts())

    with col2:
        st.caption("Materiality Distribution")
        if "overall_materiality" in filtered.columns:
            st.bar_chart(filtered["overall_materiality"].value_counts())

    st.caption("Sample Risks")
    sample_cols = [
        "group_id",
        "risk_name",
        "business_division",
        "risk_type",
        "overall_materiality",
    ]
    sample_cols = [c for c in sample_cols if c in filtered.columns]

    st.dataframe(
        filtered[sample_cols].head(15),
        use_container_width=True,
    )


# ======================================================
# Chat tab
# ======================================================

with tab_chat:
    st.markdown("#### Ask about the filtered risk inventory")

    st.caption(
        "Examples: summarize material risks, compare QoQ changes, review taxonomy mapping, prepare executive brief."
    )

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {
                "role": "assistant",
                "content": "Ask about the filtered risk inventory. I will use the current filters and cite group_id.",
            }
        ]

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask about risks...")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        cfg = {
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "api_version": os.getenv("OPENAI_API_VERSION", "2024-10-21"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        }

        missing = [k for k, v in cfg.items() if not v and k != "api_key"]

        if missing:
            answer = f"Missing Azure OpenAI config: {', '.join(missing)}"
        else:
            with st.spinner("Analyzing inventory..."):
                data_context = build_context(filtered, prompt, max_rows=max_rows)

                skill_key, skill_instruction = get_skill_instruction(
                    prompt=prompt,
                    skills_dir=str(SKILLS_DIR),
                    index_path=str(SKILLS_INDEX_PATH),
                )

                if show_context:
                    with st.expander("AI Context Preview", expanded=False):
                        st.caption(f"Selected skill: {skill_key or 'None'}")
                        st.code(data_context[:8000], language="json")

                answer = ask_ai(
                    chat_history=st.session_state.chat,
                    cfg=cfg,
                    prompt=prompt,
                    data_context=data_context,
                    skill_instruction=skill_instruction,
                )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.rerun()
