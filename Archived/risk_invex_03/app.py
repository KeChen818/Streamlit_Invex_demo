from html import escape

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Risk Invex | Enterprise Dashboard",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Load CSS
# -----------------------------
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")


# -----------------------------
# Mock data
# -----------------------------
risk_data = pd.DataFrame([
    {
        "Risk": "Third-party cyber breach",
        "Entity": "Enterprise",
        "Category": "Technology",
        "Impact": "High",
        "Likelihood": "High",
        "Change": "New",
        "Owner": "Technology Risk",
        "Control Focus": "Vendor access review",
        "DQ Score": 91,
    },
    {
        "Risk": "AI model bias compliance",
        "Entity": "Consumer Banking",
        "Category": "Compliance",
        "Impact": "High",
        "Likelihood": "Medium",
        "Change": "Increased",
        "Owner": "Compliance",
        "Control Focus": "Model validation and fairness testing",
        "DQ Score": 89,
    },
    {
        "Risk": "Vendor concentration",
        "Entity": "Payments",
        "Category": "Operational",
        "Impact": "Medium",
        "Likelihood": "Medium",
        "Change": "New",
        "Owner": "Ops Risk",
        "Control Focus": "Third-party exit planning",
        "DQ Score": 87,
    },
    {
        "Risk": "Cloud cost escalation",
        "Entity": "Enterprise",
        "Category": "Financial",
        "Impact": "Medium",
        "Likelihood": "High",
        "Change": "Increased",
        "Owner": "Finance",
        "Control Focus": "Cloud usage thresholds",
        "DQ Score": 94,
    },
    {
        "Risk": "Data quality degradation",
        "Entity": "Wealth",
        "Category": "Operational",
        "Impact": "Medium",
        "Likelihood": "Medium",
        "Change": "Decreased",
        "Owner": "Data Governance",
        "Control Focus": "Data lineage controls",
        "DQ Score": 96,
    },
    {
        "Risk": "Liquidity stress in volatile markets",
        "Entity": "Markets",
        "Category": "Strategic",
        "Impact": "High",
        "Likelihood": "Medium",
        "Change": "Stable",
        "Owner": "Strategy",
        "Control Focus": "Liquidity scenario monitoring",
        "DQ Score": 90,
    },
    {
        "Risk": "Commercial real estate exposure",
        "Entity": "Commercial Banking",
        "Category": "Credit",
        "Impact": "High",
        "Likelihood": "Medium",
        "Change": "Increased",
        "Owner": "Credit Risk",
        "Control Focus": "CRE exposure review",
        "DQ Score": 88,
    },
    {
        "Risk": "AI governance attestation gap",
        "Entity": "Enterprise",
        "Category": "Regulatory",
        "Impact": "High",
        "Likelihood": "Low",
        "Change": "New",
        "Owner": "Regulatory Affairs",
        "Control Focus": "AI governance attestation",
        "DQ Score": 92,
    },
    {
        "Risk": "Synthetic identity fraud",
        "Entity": "Payments",
        "Category": "Fraud",
        "Impact": "Medium",
        "Likelihood": "High",
        "Change": "Increased",
        "Owner": "Fraud Risk",
        "Control Focus": "Transaction anomaly monitoring",
        "DQ Score": 93,
    },
    {
        "Risk": "Privileged access drift",
        "Entity": "Wealth",
        "Category": "Technology",
        "Impact": "High",
        "Likelihood": "Low",
        "Change": "Stable",
        "Owner": "Technology Risk",
        "Control Focus": "Privileged access review",
        "DQ Score": 95,
    },
    {
        "Risk": "Contact center continuity",
        "Entity": "Consumer Banking",
        "Category": "Operational",
        "Impact": "Low",
        "Likelihood": "Medium",
        "Change": "Resolved",
        "Owner": "Ops Risk",
        "Control Focus": "Continuity test evidence",
        "DQ Score": 98,
    },
    {
        "Risk": "Funding concentration",
        "Entity": "Markets",
        "Category": "Financial",
        "Impact": "Medium",
        "Likelihood": "Medium",
        "Change": "Decreased",
        "Owner": "Treasury",
        "Control Focus": "Funding concentration limits",
        "DQ Score": 97,
    },
])

RISK_LEVELS = ["Low", "Medium", "High"]
RISK_LEVEL_SCORE = {level: idx for idx, level in enumerate(RISK_LEVELS, start=1)}
CHANGE_ORDER = ["New", "Increased", "Stable", "Decreased", "Resolved"]
CHAT_SUGGESTIONS = [
    "Summarize this risk slice",
    "Which risks are most material?",
    "What changed this quarter?",
    "Which controls need review?",
]
NAV_SECTIONS = {
    "Dashboard": ["Home", "Risk Heatmap", "QoQ Changes"],
    "Inventory": ["Risk Library", "Controls", "Issues"],
    "Intelligence": ["AI Risk Analyst", "Scenario Analysis"],
}
PAGE_TITLES = {
    "Home": "Risk Inventory Intelligence",
    "Risk Heatmap": "Risk Heatmap",
    "QoQ Changes": "Quarter-over-Quarter Changes",
    "Risk Library": "Risk Library",
    "Controls": "Controls Coverage",
    "Issues": "Issues and Exceptions",
    "AI Risk Analyst": "AI Risk Analyst",
    "Scenario Analysis": "Scenario Analysis",
}


# -----------------------------
# Helpers
# -----------------------------
def sorted_options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().unique().tolist())


def apply_slicers(
    data: pd.DataFrame,
    entities: list[str],
    categories: list[str],
    impacts: list[str],
    likelihoods: list[str],
    changes: list[str],
    search_term: str,
) -> pd.DataFrame:
    filtered = data.copy()
    slicer_map = {
        "Entity": entities,
        "Category": categories,
        "Impact": impacts,
        "Likelihood": likelihoods,
        "Change": changes,
    }

    for column, selected_values in slicer_map.items():
        if selected_values:
            filtered = filtered[filtered[column].isin(selected_values)]

    clean_search = search_term.strip().lower()
    if clean_search:
        search_columns = ["Risk", "Entity", "Category", "Owner", "Control Focus", "Change"]
        searchable_text = filtered[search_columns].astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[searchable_text.str.contains(clean_search, regex=False, na=False)]

    return filtered


def rank_risks(data: pd.DataFrame) -> pd.DataFrame:
    ranked = data.copy()
    ranked["_score"] = (
        ranked["Impact"].map(RISK_LEVEL_SCORE).fillna(0) * 10
        + ranked["Likelihood"].map(RISK_LEVEL_SCORE).fillna(0)
    )
    return ranked.sort_values(["_score", "DQ Score"], ascending=[False, True])


def format_counts(series: pd.Series, limit: int = 3) -> str:
    counts = series.value_counts().head(limit)
    if counts.empty:
        return "none"
    return ", ".join(f"{label} ({count})" for label, count in counts.items())


def build_heatmap_points(data: pd.DataFrame) -> tuple[list[list[int]], int]:
    points = []
    values = []
    for likelihood_idx, likelihood in enumerate(RISK_LEVELS):
        for impact_idx, impact in enumerate(RISK_LEVELS):
            count = int(
                ((data["Impact"] == impact) & (data["Likelihood"] == likelihood)).sum()
            )
            points.append([impact_idx, likelihood_idx, count])
            values.append(count)
    return points, max(values + [1])


def top_category_data(data: pd.DataFrame) -> tuple[list[str], list[int]]:
    if data.empty:
        return ["No matching risks"], [0]

    counts = data["Category"].value_counts().head(5)
    return counts.index.tolist(), counts.values.tolist()


def build_ai_reply(prompt: str, data: pd.DataFrame) -> str:
    clean_prompt = prompt.strip()
    lower_prompt = clean_prompt.lower()

    if data.empty:
        return (
            "No risks match the current slicers. Broaden the slice and I can summarize "
            "material risks, owner concentration, changes, and control focus."
        )

    total = len(data)
    ranked = rank_risks(data).head(4)
    top_categories = format_counts(data["Category"])
    top_owners = format_counts(data["Owner"])
    high_count = int((data["Impact"] == "High").sum())
    increased_count = int((data["Change"] == "Increased").sum())
    new_count = int((data["Change"] == "New").sum())
    avg_dq = data["DQ Score"].mean()

    if any(token in lower_prompt for token in ["owner", "approval", "accountable", "who"]):
        return (
            f"Ownership is concentrated in {top_owners}. The current slice has {total} "
            f"risks, with {high_count} high-impact items. Review accountability first "
            "where the same owner appears across multiple high-impact risks."
        )

    if any(token in lower_prompt for token in ["change", "quarter", "qoq", "new", "resolved"]):
        change_summary = format_counts(data["Change"], limit=5)
        return (
            f"Quarterly movement in this slice: {change_summary}. The main watch items "
            f"are {new_count} new risks and {increased_count} increased risks. Prioritize "
            "fresh entries before stable or resolved records."
        )

    if any(token in lower_prompt for token in ["control", "evidence", "review", "mitigation"]):
        control_lines = [
            f"- {row['Risk']}: {row['Control Focus']} ({row['Owner']})"
            for _, row in ranked.iterrows()
        ]
        return "Recommended control review queue:\n\n" + "\n".join(control_lines)

    if any(token in lower_prompt for token in ["high", "material", "highest", "severe"]):
        risk_lines = [
            f"- {row['Risk']}: {row['Impact']} impact / {row['Likelihood']} likelihood"
            for _, row in ranked.iterrows()
        ]
        return "Most material risks in the current slice:\n\n" + "\n".join(risk_lines)

    return (
        f"The current slice contains {total} risks across {top_categories}. "
        f"{high_count} are high-impact, average DQ confidence is {avg_dq:.0f}%, "
        f"and ownership is led by {top_owners}. I would focus first on new or "
        "increased risks with high impact and any control evidence below the pack."
    )


def add_chat_turn(prompt: str, data: pd.DataFrame) -> None:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        return

    st.session_state.risk_chat_messages.append({"role": "user", "content": clean_prompt})
    st.session_state.risk_chat_messages.append({
        "role": "assistant",
        "content": build_ai_reply(clean_prompt, data),
    })


def render_alert_rows(data: pd.DataFrame) -> str:
    if data.empty:
        return "<div class='empty-state tight'>No matching risks in current slicer.</div>"

    rows = []
    for _, row in rank_risks(data).head(4).iterrows():
        tone = "risk-high" if row["Impact"] == "High" else "risk-med"
        rows.append(
            "<div class='alert-row'>"
            f"<span>{escape(row['Risk'])}</span>"
            f"<b class='{tone}'>{escape(row['Impact'])}</b>"
            "</div>"
        )
    return "\n".join(rows)


def render_recent_rows(data: pd.DataFrame) -> str:
    if data.empty:
        return "<div class='empty-state tight'>No matching risk changes.</div>"

    changed = data[data["Change"].isin(["New", "Increased", "Decreased", "Resolved"])].copy()
    changed["_change_rank"] = changed["Change"].apply(lambda item: CHANGE_ORDER.index(item))
    changed = changed.sort_values("_change_rank").head(3)

    rows = []
    for _, row in changed.iterrows():
        rows.append(
            "<div class='alert-row'>"
            f"<span>{escape(row['Risk'])}</span>"
            f"<em>{escape(row['Change'])}</em>"
            "</div>"
        )
    return "\n".join(rows)


if "risk_chat_messages" not in st.session_state:
    st.session_state.risk_chat_messages = [{
        "role": "assistant",
        "content": (
            "Ask me about the current slicer context. I can summarize exposure, "
            "rank material risks, explain quarter-over-quarter movement, or build a "
            "control review queue."
        ),
    }]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div class='brand'>RISK INVEX</div>", unsafe_allow_html=True)
    st.markdown("<div class='go-button'>&lt; GO &gt;</div>", unsafe_allow_html=True)

    st.markdown("### Dashboard")
    st.markdown("Executive Summary")
    st.markdown("Risk Heatmap")
    st.markdown("QoQ Changes")

    st.markdown("### Inventory")
    st.markdown("Risk Library")
    st.markdown("Controls")
    st.markdown("Issues")

    st.markdown("### Intelligence")
    st.markdown("AI Risk Analyst")
    st.markdown("Scenario Analysis")


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="top-header">
  <div>
    <div class="eyebrow">Enterprise Risk Command Center</div>
    <div class="title">Risk Inventory Intelligence</div>
  </div>
  <div class="header-pill">Q2 2026 Active Cycle</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Slicers
# -----------------------------
st.markdown("""
<div class="slicer-shell">
  <div>
    <div class="slicer-kicker">Risk Slicers</div>
    <div class="slicer-title">Filter inventory by entity, category, severity, movement, or keyword.</div>
  </div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3, s4, s5, s6 = st.columns([1.15, 1.15, 0.9, 0.9, 1.05, 1.35])

with s1:
    selected_entities = st.multiselect(
        "Entity",
        sorted_options(risk_data["Entity"]),
        placeholder="All entities",
    )

with s2:
    selected_categories = st.multiselect(
        "Category",
        sorted_options(risk_data["Category"]),
        placeholder="All categories",
    )

with s3:
    selected_impacts = st.multiselect(
        "Impact",
        RISK_LEVELS,
        placeholder="All impacts",
    )

with s4:
    selected_likelihoods = st.multiselect(
        "Likelihood",
        RISK_LEVELS,
        placeholder="All likelihoods",
    )

with s5:
    selected_changes = st.multiselect(
        "Change",
        CHANGE_ORDER,
        placeholder="All changes",
    )

with s6:
    search_term = st.text_input(
        "Search",
        placeholder="Risk, owner, control",
    )

filtered_data = apply_slicers(
    risk_data,
    selected_entities,
    selected_categories,
    selected_impacts,
    selected_likelihoods,
    selected_changes,
    search_term,
)

active_filter_count = sum(
    bool(value)
    for value in [
        selected_entities,
        selected_categories,
        selected_impacts,
        selected_likelihoods,
        selected_changes,
        search_term.strip(),
    ]
)
filter_note = "All inventory" if active_filter_count == 0 else f"{active_filter_count} active slicers"

st.markdown(
    f"""
    <div class="filter-summary">
      <span>{len(filtered_data)} of {len(risk_data)} risks shown</span>
      <b>{filter_note}</b>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# KPI cards
# -----------------------------
high_risks = int((filtered_data["Impact"] == "High").sum())
high_likelihood = int((filtered_data["Likelihood"] == "High").sum())
qoq_changes = int(filtered_data["Change"].isin(["New", "Increased", "Decreased"]).sum())
new_count = int((filtered_data["Change"] == "New").sum())
resolved_count = int((filtered_data["Change"] == "Resolved").sum())
low_dq_count = int((filtered_data["DQ Score"] < 90).sum())
dq_value = "n/a" if filtered_data.empty else f"{filtered_data['DQ Score'].mean():.0f}%"

c1, c2, c3, c4 = st.columns(4)
kpis = [
    ("Filtered Risks", str(len(filtered_data)), filter_note, "neutral"),
    ("High Impact", str(high_risks), f"{high_likelihood} high-likelihood items", "down"),
    ("QoQ Changes", str(qoq_changes), f"{new_count} new / {resolved_count} resolved", "neutral"),
    ("DQ Score", dq_value, f"{low_dq_count} records below 90%", "up" if low_dq_count == 0 else "down"),
]

for col, (label, value, foot, cls) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot {cls}">{foot}</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------
# Charts
# -----------------------------
left, right = st.columns([1.35, 0.85])

with left:
    chart_a, chart_b = st.columns(2)
    heatmap_points, heatmap_max = build_heatmap_points(filtered_data)
    category_labels, category_values = top_category_data(filtered_data)

    with chart_a:
        st.markdown("<div class='section-title'>Risk Heatmap</div>", unsafe_allow_html=True)
        heatmap_option = {
            "tooltip": {},
            "grid": {"top": 35, "right": 20, "bottom": 40, "left": 65},
            "xAxis": {
                "type": "category",
                "name": "Impact",
                "data": RISK_LEVELS,
                "axisLabel": {"color": "#9ca3af"},
                "nameTextStyle": {"color": "#9ca3af"},
            },
            "yAxis": {
                "type": "category",
                "name": "Likelihood",
                "data": RISK_LEVELS,
                "axisLabel": {"color": "#9ca3af"},
                "nameTextStyle": {"color": "#9ca3af"},
            },
            "visualMap": {
                "min": 0,
                "max": heatmap_max,
                "orient": "horizontal",
                "left": "center",
                "bottom": 0,
                "textStyle": {"color": "#9ca3af"},
                "inRange": {"color": ["#14532d", "#eab308", "#dc2626"]},
            },
            "series": [{
                "type": "heatmap",
                "data": heatmap_points,
                "label": {"show": True, "color": "#ffffff"},
            }],
        }
        st_echarts(heatmap_option, height="340px")

    with chart_b:
        st.markdown("<div class='section-title'>Top Risk Categories</div>", unsafe_allow_html=True)
        bar_option = {
            "tooltip": {},
            "grid": {"top": 20, "right": 25, "bottom": 25, "left": 115},
            "xAxis": {"type": "value", "axisLabel": {"color": "#9ca3af"}},
            "yAxis": {
                "type": "category",
                "inverse": True,
                "data": category_labels,
                "axisLabel": {"color": "#9ca3af"},
            },
            "series": [{
                "type": "bar",
                "data": category_values,
                "barWidth": 16,
                "itemStyle": {"borderRadius": [0, 8, 8, 0], "color": "#f59e0b"},
                "label": {"show": True, "position": "right", "color": "#f5f5f5"},
            }]
        }
        st_echarts(bar_option, height="340px")

    st.markdown("<div class='section-title'>Latest Risk Changes</div>", unsafe_allow_html=True)
    display_columns = [
        "Risk",
        "Entity",
        "Category",
        "Impact",
        "Likelihood",
        "Change",
        "Owner",
        "Control Focus",
        "DQ Score",
    ]

    if filtered_data.empty:
        st.markdown(
            "<div class='empty-state'>No risk records match the current slicers.</div>",
            unsafe_allow_html=True,
        )
    else:
        ranked_display = rank_risks(filtered_data).drop(columns=["_score"])
        st.dataframe(ranked_display[display_columns], width="stretch", hide_index=True)

with right:
    st.markdown("<div class='section-title'>AI Risk Analyst Chat</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="chat-context">
          Context: {len(filtered_data)} filtered risks | {format_counts(filtered_data['Category'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    clear_col, context_col = st.columns([0.8, 1.2])
    with clear_col:
        if st.button("Clear chat", key="clear_risk_chat", width="stretch"):
            st.session_state.risk_chat_messages = [{
                "role": "assistant",
                "content": "Chat cleared. Ask me about the current slicer context.",
            }]
            st.rerun()
    with context_col:
        st.markdown(
            f"<div class='chat-status'>{filter_note}</div>",
            unsafe_allow_html=True,
        )

    for message in st.session_state.risk_chat_messages[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    suggestion_cols = st.columns(2)
    for idx, suggestion in enumerate(CHAT_SUGGESTIONS):
        with suggestion_cols[idx % 2]:
            if st.button(suggestion, key=f"chat_suggestion_{idx}", width="stretch"):
                add_chat_turn(suggestion, filtered_data)
                st.rerun()

    with st.form("risk_ai_chat_form", clear_on_submit=True):
        user_prompt = st.text_area(
            "Ask the AI risk analyst",
            placeholder="Ask about exposure, owners, controls, changes...",
            height=78,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", width="stretch")

    if submitted and user_prompt.strip():
        add_chat_turn(user_prompt, filtered_data)
        st.rerun()

    st.markdown(
        f"""
        <div class="insight-card">
          <div class="section-title">Emerging Risks</div>
          {render_alert_rows(filtered_data)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="insight-card">
          <div class="section-title">Recent Risk Changes</div>
          {render_recent_rows(filtered_data)}
        </div>
        """,
        unsafe_allow_html=True,
    )
