
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_TITLE = "RISKEX — Risk Inventory Intelligence"
DATA_PATH = Path(__file__).parent / "sample_risk_inventory.json"


st.set_page_config(
    page_title="RISKEX",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
}

[data-testid="stSidebar"] {
  background: #0b1120;
  border-right: 1px solid #334155;
}

.riskex-hero {
  padding: 1.2rem 1.4rem;
  border: 1px solid #334155;
  border-radius: 18px;
  background: rgba(17, 24, 39, 0.85);
  margin-bottom: 1rem;
}

.riskex-title {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  margin-bottom: 0.2rem;
  color: #f8fafc;
}

.riskex-subtitle {
  color: #94a3b8;
  font-size: 0.95rem;
}

.metric-card {
  padding: 1rem;
  border: 1px solid #334155;
  border-radius: 16px;
  background: rgba(31, 41, 55, 0.85);
}

.metric-label {
  color: #94a3b8;
  font-size: 0.8rem;
}

.metric-value {
  font-size: 1.6rem;
  font-weight: 800;
  margin-top: 0.2rem;
  color: #f8fafc;
}

.section-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin: 1rem 0 0.5rem 0;
  color: #f8fafc;
}

.badge-high {
  color: #fee2e2;
  background: rgba(239, 68, 68, 0.18);
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
}

.badge-medium {
  color: #fef3c7;
  background: rgba(245, 158, 11, 0.18);
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
}

.badge-low {
  color: #dcfce7;
  background: rgba(34, 197, 94, 0.18);
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
}

div[data-testid="stChatMessage"] {
  background: rgba(31, 41, 55, 0.65);
  border: 1px solid rgba(51, 65, 85, 0.8);
  border-radius: 14px;
}
</style>
"""


def load_data() -> pd.DataFrame:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def risk_score(row):
    likelihood_map = {"Low": 1, "Medium": 2, "High": 3}
    impact_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    return likelihood_map.get(row.get("likelihood"), 1) * impact_map.get(row.get("impact"), 1)


def build_context(df: pd.DataFrame) -> str:
    keep_cols = [
        "group_id", "business_division", "legal_entity", "risk_category",
        "risk_name", "risk_description", "likelihood", "impact", "status", "owner"
    ]
    records = df[keep_cols].to_dict(orient="records")
    return json.dumps(records[:30], ensure_ascii=False, indent=2)


def ask_ai(messages, scoped_df, sidebar_key=""):
    api_key = sidebar_key or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return (
            "Demo mode: OpenAI API key is not configured. "
            "Based on the filtered inventory, focus on High/Critical impact risks, "
            "pending approval items, and risks with repeated categories or owners. "
            "Set OPENAI_API_KEY to enable live AI analysis."
        )

    client = OpenAI(api_key=api_key)
    context = build_context(scoped_df)

    system_prompt = f"""
You are a risk inventory intelligence assistant.
Only answer based on the provided risk inventory context.
Do not invent records, ratings, owners, or numbers.
When referring to a specific risk, cite its group_id.

Risk inventory context:
{context}
"""

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.2"),
        input=[
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    )
    return response.output_text


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

df = load_data()
df["risk_score"] = df.apply(risk_score, axis=1)

with st.sidebar:
    st.markdown("## ⚠️ RISKEX")
    st.caption("Risk Inventory Intelligence")
    st.divider()

    divisions = ["All"] + sorted(df["business_division"].unique().tolist())
    entities = ["All"] + sorted(df["legal_entity"].unique().tolist())
    categories = ["All"] + sorted(df["risk_category"].unique().tolist())
    impacts = ["All"] + sorted(df["impact"].unique().tolist())

    selected_division = st.selectbox("Business Division", divisions)
    selected_entity = st.selectbox("Legal Entity", entities)
    selected_category = st.selectbox("Risk Category", categories)
    selected_impact = st.selectbox("Impact", impacts)
    search = st.text_input("Search risk name / description")

    st.divider()
    st.markdown("**AI Configuration**")

    if "api_key_confirmed" not in st.session_state:
        st.session_state.api_key_confirmed = False
    if "api_key_value" not in st.session_state:
        st.session_state.api_key_value = os.getenv("OPENAI_API_KEY", "")

    sidebar_api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key to enable AI chat.",
        value="",
    )

    if st.button("Confirm API Key", use_container_width=True):
        key_to_test = sidebar_api_key_input or os.getenv("OPENAI_API_KEY", "")
        if not key_to_test:
            st.session_state.api_key_confirmed = False
            st.warning("Please enter an API key.")
        elif OpenAI is None:
            st.session_state.api_key_confirmed = False
            st.error("OpenAI library not installed.")
        else:
            try:
                test_client = OpenAI(api_key=key_to_test)
                test_client.models.list()
                st.session_state.api_key_value = key_to_test
                st.session_state.api_key_confirmed = True
            except Exception:
                st.session_state.api_key_confirmed = False
                st.error("Invalid or unauthorized API key.")

    if st.session_state.api_key_confirmed:
        st.success("API key confirmed — AI chat enabled.")
    elif st.session_state.api_key_value:
        st.info("API key loaded from environment.")

    sidebar_api_key = st.session_state.api_key_value
    st.caption("Your key is only used in this session and never stored.")


filtered = df.copy()

if selected_division != "All":
    filtered = filtered[filtered["business_division"] == selected_division]
if selected_entity != "All":
    filtered = filtered[filtered["legal_entity"] == selected_entity]
if selected_category != "All":
    filtered = filtered[filtered["risk_category"] == selected_category]
if selected_impact != "All":
    filtered = filtered[filtered["impact"] == selected_impact]
if search:
    s = search.lower()
    filtered = filtered[
        filtered["risk_name"].str.lower().str.contains(s)
        | filtered["risk_description"].str.lower().str.contains(s)
    ]


st.markdown(
    """
    <div class="riskex-hero">
      <div class="riskex-title">RISKEX — Risk Inventory Intelligence</div>
      <div class="riskex-subtitle">
        Streamlit prototype: inventory browser, analytics, and AI-assisted risk review.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Risks</div><div class="metric-value">{len(filtered)}</div></div>', unsafe_allow_html=True)
with m2:
    high_count = len(filtered[filtered["impact"].isin(["High", "Critical"])])
    st.markdown(f'<div class="metric-card"><div class="metric-label">High / Critical</div><div class="metric-value">{high_count}</div></div>', unsafe_allow_html=True)
with m3:
    pending_count = len(filtered[filtered["status"].str.contains("Pending", case=False)])
    st.markdown(f'<div class="metric-card"><div class="metric-label">Pending Approval</div><div class="metric-value">{pending_count}</div></div>', unsafe_allow_html=True)
with m4:
    avg_score = round(filtered["risk_score"].mean(), 2) if len(filtered) else 0
    st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Risk Score</div><div class="metric-value">{avg_score}</div></div>', unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["Inventory", "Analytics", "AI Chat"])

with tab1:
    st.markdown('<div class="section-title">Risk Inventory</div>', unsafe_allow_html=True)
    display_cols = [
        "group_id", "business_division", "legal_entity", "risk_category",
        "risk_name", "likelihood", "impact", "status", "owner", "risk_score"
    ]
    st.dataframe(
        filtered[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        csv,
        file_name="filtered_risk_inventory.csv",
        mime="text/csv",
    )

with tab2:
    st.markdown('<div class="section-title">Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.caption("Risks by category")
        chart_df = filtered.groupby("risk_category").size().reset_index(name="count")
        st.bar_chart(chart_df, x="risk_category", y="count", use_container_width=True)

    with c2:
        st.caption("Risks by business division")
        chart_df = filtered.groupby("business_division").size().reset_index(name="count")
        st.bar_chart(chart_df, x="business_division", y="count", use_container_width=True)

    st.caption("Top risk score items")
    top_df = filtered.sort_values("risk_score", ascending=False).head(8)
    st.dataframe(
        top_df[["group_id", "risk_name", "impact", "likelihood", "risk_score", "owner"]],
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.markdown('<div class="section-title">AI Chat</div>', unsafe_allow_html=True)
    st.caption("Ask questions about the currently filtered inventory scope.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Ask me about the selected risk inventory scope."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask about key risks, high impact items, owners, or pending approvals...")

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing inventory..."):
                ai_input = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_messages
                    if m["role"] in ["user", "assistant"]
                ]
                answer = ask_ai(ai_input, filtered, sidebar_key=sidebar_api_key)
                st.write(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
