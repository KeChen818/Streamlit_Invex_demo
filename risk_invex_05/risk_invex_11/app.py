from html import escape

import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

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

OWNER_TYPE_BY_OWNER = {
    "Technology Risk": "Second Line",
    "Compliance": "Second Line",
    "Ops Risk": "First Line",
    "Finance": "First Line",
    "Data Governance": "Second Line",
    "Strategy": "First Line",
    "Credit Risk": "Second Line",
    "Regulatory Affairs": "Second Line",
    "Fraud Risk": "Second Line",
    "Treasury": "First Line",
}
risk_data["Owner Type"] = risk_data["Owner"].map(OWNER_TYPE_BY_OWNER).fillna("Unassigned")

RISK_LEVELS = ["Very Low", "Low", "Medium", "High", "Very High"]
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
        search_columns = ["Risk", "Entity", "Category", "Owner", "Owner Type", "Control Focus", "Change"]
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


def is_material_cell(impact: str, likelihood: str) -> bool:
    impact_score = RISK_LEVEL_SCORE.get(impact, 0)
    likelihood_score = RISK_LEVEL_SCORE.get(likelihood, 0)
    return impact_score * likelihood_score >= 12


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


def donut_data(data: pd.DataFrame, column: str, limit: int = 5) -> list[dict[str, object]]:
    if data.empty:
        return [{"name": "No data", "value": 0}]

    counts = data[column].value_counts().head(limit)
    return [{"name": label, "value": int(value)} for label, value in counts.items()]


def apply_home_ring_filters(data: pd.DataFrame, filters: dict[str, str | None]) -> pd.DataFrame:
    filtered = data.copy()
    for column, selected_value in filters.items():
        if selected_value:
            filtered = filtered[filtered[column] == selected_value]
    return filtered


def apply_home_heatmap_filter(
    data: pd.DataFrame,
    selected_cell: tuple[str, str] | None,
) -> pd.DataFrame:
    if not selected_cell:
        return data

    impact, likelihood = selected_cell
    return data[(data["Impact"] == impact) & (data["Likelihood"] == likelihood)]


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
    high_count = int(data["Impact"].isin(["High", "Very High"]).sum())
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
        tone = "risk-high" if row["Impact"] in ["High", "Very High"] else "risk-med"
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


def metric_values(data: pd.DataFrame, filter_note: str) -> list[tuple[str, str, str, str]]:
    high_risks = int(data["Impact"].isin(["High", "Very High"]).sum())
    high_likelihood = int(data["Likelihood"].isin(["High", "Very High"]).sum())
    qoq_changes = int(data["Change"].isin(["New", "Increased", "Decreased"]).sum())
    new_count = int((data["Change"] == "New").sum())
    resolved_count = int((data["Change"] == "Resolved").sum())
    low_dq_count = int((data["DQ Score"] < 90).sum())
    dq_value = "n/a" if data.empty else f"{data['DQ Score'].mean():.0f}%"

    return [
        ("Filtered Risks", str(len(data)), filter_note, "neutral"),
        ("High Impact", str(high_risks), f"{high_likelihood} high-likelihood items", "down"),
        ("QoQ Changes", str(qoq_changes), f"{new_count} new / {resolved_count} resolved", "neutral"),
        ("DQ Score", dq_value, f"{low_dq_count} records below 90%", "up" if low_dq_count == 0 else "down"),
    ]


def render_metric_cards(data: pd.DataFrame, filter_note: str) -> None:
    c1, c2, c3, c4 = st.columns(4)

    for col, (label, value, foot, cls) in zip([c1, c2, c3, c4], metric_values(data, filter_note)):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-foot {cls}">{foot}</div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_table(data: pd.DataFrame, columns: list[str] | None = None) -> None:
    display_columns = columns or [
        "Risk",
        "Entity",
        "Category",
        "Impact",
        "Likelihood",
        "Change",
        "Owner",
        "Owner Type",
        "Control Focus",
        "DQ Score",
    ]

    if data.empty:
        st.markdown(
            "<div class='empty-state'>No risk records match the current slicers.</div>",
            unsafe_allow_html=True,
        )
        return

    ranked_display = rank_risks(data).drop(columns=["_score"])
    st.dataframe(ranked_display[display_columns], width="stretch", hide_index=True)


def render_heatmap_chart(
    data: pd.DataFrame,
    height: str = "340px",
    key: str | None = None,
    selected_cell: tuple[str, str] | None = None,
    selectable: bool = False,
):
    heatmap_points, _ = build_heatmap_points(data)
    selected_indices = None
    if selected_cell:
        impact, likelihood = selected_cell
        if impact in RISK_LEVELS and likelihood in RISK_LEVELS:
            selected_indices = (RISK_LEVELS.index(impact), RISK_LEVELS.index(likelihood))

    series_data = []
    for point in heatmap_points:
        impact = RISK_LEVELS[point[0]]
        likelihood = RISK_LEVELS[point[1]]
        material = is_material_cell(impact, likelihood)
        item_style = {
            "color": "#cf2e2e" if material else "#343a40",
            "borderColor": "#0b0b0b",
            "borderWidth": 1,
        }
        if selected_indices and point[0] == selected_indices[0] and point[1] == selected_indices[1]:
            item_style.update({
                "borderColor": "#f5f5f5",
                "borderWidth": 2,
                "shadowBlur": 8,
                "shadowColor": "rgba(245, 164, 0, 0.45)",
            })
        cell = {
            "value": point,
            "itemStyle": item_style,
        }
        series_data.append(cell)

    heatmap_option = {
        "tooltip": {
            "formatter": JsCode(
                "function(params) {"
                "var levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];"
                "var impact = levels[params.value[0]];"
                "var likelihood = levels[params.value[1]];"
                "return params.marker + impact + ' impact / ' + likelihood + ' likelihood: ' + params.value[2] + ' risks';"
                "}"
            ).js_code
        },
        "grid": {"top": 35, "right": 20, "bottom": 35, "left": 82},
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
        "series": [{
            "type": "heatmap",
            "data": series_data,
            "label": {"show": True, "color": "#ffffff", "fontWeight": 800},
            "emphasis": {"itemStyle": {"borderColor": "#f5a400", "borderWidth": 2}},
        }],
    }
    events = None
    if selectable:
        events = {
            "click": (
                "function(params) {"
                "if (!params || !params.value) { return null; }"
                "return {impactIndex: params.value[0], likelihoodIndex: params.value[1], count: params.value[2]};"
                "}"
            )
        }
    return st_echarts(heatmap_option, height=height, events=events, key=key)


def render_materiality_legend() -> None:
    st.markdown(
        """
        <div class="matrix-legend">
          <span><i class="material"></i>Material</span>
          <span><i class="non-material"></i>Non-material</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_donut_chart(
    data: pd.DataFrame,
    column: str,
    key: str,
    height: str = "175px",
    selected_value: str | None = None,
    selectable: bool = False,
):
    chart_data = donut_data(data, column)
    for item in chart_data:
        if selected_value and item["name"] == selected_value:
            item["selected"] = True
            item["itemStyle"] = {
                "borderColor": "#f5f5f5",
                "borderWidth": 2,
                "shadowBlur": 8,
                "shadowColor": "rgba(245, 164, 0, 0.45)",
            }

    option = {
        "color": ["#f5a400", "#cf2e2e", "#64748b", "#22c55e", "#38bdf8", "#a78bfa"],
        "tooltip": {"trigger": "item"},
        "legend": {
            "type": "scroll",
            "orient": "vertical",
            "right": 0,
            "top": "middle",
            "itemWidth": 8,
            "itemHeight": 8,
            "textStyle": {"color": "#9ca3af", "fontSize": 10},
        },
        "series": [{
            "type": "pie",
            "selectedMode": "single",
            "selectedOffset": 5,
            "radius": ["48%", "72%"],
            "center": ["34%", "50%"],
            "avoidLabelOverlap": True,
            "label": {"show": False},
            "labelLine": {"show": False},
            "data": chart_data,
        }],
    }
    events = None
    if selectable:
        events = {
            "click": (
                "function(params) {"
                "if (!params || !params.name || params.name === 'No data') { return null; }"
                f"return {{column: '{column}', name: params.name, value: params.value}};"
                "}"
            )
        }

    return st_echarts(option, height=height, key=key, events=events)


def render_category_chart(data: pd.DataFrame, height: str = "340px") -> None:
    category_labels, category_values = top_category_data(data)
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
    st_echarts(bar_option, height=height)


def render_chat_panel(data: pd.DataFrame, filter_note: str, key_prefix: str) -> None:
    st.markdown("<div class='section-title'>AI Risk Analyst Chat</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="chat-context">
          Context: {len(data)} filtered risks | {format_counts(data['Category'])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    clear_col, context_col = st.columns([0.8, 1.2])
    with clear_col:
        if st.button("Clear chat", key=f"clear_risk_chat_{key_prefix}", width="stretch"):
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

    for message in st.session_state.risk_chat_messages[-8:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    suggestion_cols = st.columns(2)
    for idx, suggestion in enumerate(CHAT_SUGGESTIONS):
        with suggestion_cols[idx % 2]:
            if st.button(suggestion, key=f"chat_suggestion_{key_prefix}_{idx}", width="stretch"):
                add_chat_turn(suggestion, data)
                st.rerun()

    with st.form(f"risk_ai_chat_form_{key_prefix}", clear_on_submit=True):
        user_prompt = st.text_area(
            "Ask the AI risk analyst",
            placeholder="Ask about exposure, owners, controls, changes...",
            height=78,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send", width="stretch")

    if submitted and user_prompt.strip():
        add_chat_turn(user_prompt, data)
        st.rerun()


def render_home_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    left, right = st.columns([1.35, 0.85])

    with left:
        selected_cell = st.session_state.home_heatmap_cell
        ring_filters = st.session_state.home_ring_filters.copy()
        for column, selected_value in ring_filters.items():
            if selected_value and selected_value not in data[column].dropna().unique():
                ring_filters[column] = None
        st.session_state.home_ring_filters = ring_filters

        heatmap_filtered_data = apply_home_heatmap_filter(data, selected_cell)
        category_chart_data = apply_home_ring_filters(
            heatmap_filtered_data,
            {"Owner Type": ring_filters["Owner Type"]},
        )
        owner_chart_data = apply_home_ring_filters(
            heatmap_filtered_data,
            {"Category": ring_filters["Category"]},
        )
        ring_filtered_data = apply_home_ring_filters(data, ring_filters)

        donut_col, heatmap_col = st.columns([1, 2])

        with donut_col:
            st.markdown("<div class='section-title'>Category</div>", unsafe_allow_html=True)
            clicked_category = render_donut_chart(
                category_chart_data,
                "Category",
                key=f"home_category_donut_{st.session_state.home_ring_revision}",
                selected_value=ring_filters["Category"],
                selectable=True,
            )
            st.markdown("<div class='section-title'>Owner Type</div>", unsafe_allow_html=True)
            clicked_owner_type = render_donut_chart(
                owner_chart_data,
                "Owner Type",
                key=f"home_owner_type_donut_{st.session_state.home_ring_revision}",
                selected_value=ring_filters["Owner Type"],
                selectable=True,
            )

        for clicked_ring in [clicked_category, clicked_owner_type]:
            if isinstance(clicked_ring, dict):
                column = clicked_ring.get("column")
                name = clicked_ring.get("name")
                if column in st.session_state.home_ring_filters and name:
                    current_value = st.session_state.home_ring_filters[column]
                    st.session_state.home_ring_filters[column] = None if current_value == name else name
                    st.session_state.home_heatmap_cell = None
                    st.session_state.home_ring_revision += 1
                    st.session_state.home_heatmap_revision += 1
                    st.rerun()

        with heatmap_col:
            st.markdown("<div class='section-title'>Risk Heatmap</div>", unsafe_allow_html=True)
            render_materiality_legend()
            clicked_cell = render_heatmap_chart(
                ring_filtered_data,
                height="365px",
                key=f"home_heatmap_{st.session_state.home_heatmap_revision}",
                selected_cell=selected_cell,
                selectable=True,
            )

        if isinstance(clicked_cell, dict):
            impact_index = clicked_cell.get("impactIndex")
            likelihood_index = clicked_cell.get("likelihoodIndex")
            if impact_index in range(len(RISK_LEVELS)) and likelihood_index in range(len(RISK_LEVELS)):
                next_selected_cell = (RISK_LEVELS[impact_index], RISK_LEVELS[likelihood_index])
                if next_selected_cell != selected_cell:
                    st.session_state.home_heatmap_cell = next_selected_cell
                    st.session_state.home_ring_revision += 1
                    st.rerun()
                selected_cell = next_selected_cell

        active_ring_labels = [
            f"{column}: {value}"
            for column, value in ring_filters.items()
            if value
        ]
        if active_ring_labels:
            selection_col, clear_col = st.columns([1.4, 0.45])
            with selection_col:
                st.markdown(
                    f"""
                    <div class="heatmap-selection">
                      <span>Ring Selection</span>
                      <b>{escape(' | '.join(active_ring_labels))}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with clear_col:
                if st.button("Clear rings", key="clear_home_ring_filters", width="stretch"):
                    st.session_state.home_ring_filters = {"Category": None, "Owner Type": None}
                    st.session_state.home_heatmap_cell = None
                    st.session_state.home_ring_revision += 1
                    st.session_state.home_heatmap_revision += 1
                    st.rerun()

        table_data = ring_filtered_data
        if selected_cell:
            table_data = apply_home_heatmap_filter(ring_filtered_data, selected_cell)
            impact, likelihood = selected_cell

            selection_col, clear_col = st.columns([1.4, 0.45])
            with selection_col:
                st.markdown(
                    f"""
                    <div class="heatmap-selection">
                      <span>Heatmap Selection</span>
                      <b>{escape(impact)} impact / {escape(likelihood)} likelihood</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with clear_col:
                if st.button("Clear", key="clear_home_heatmap_cell", width="stretch"):
                    st.session_state.home_heatmap_cell = None
                    st.session_state.home_heatmap_revision += 1
                    st.session_state.home_ring_revision += 1
                    st.rerun()

        st.markdown("<div class='section-title'>Inventory Table</div>", unsafe_allow_html=True)
        render_risk_table(table_data)

    with right:
        render_chat_panel(table_data, filter_note, "home")
        st.markdown(
            f"""
            <div class="insight-card">
              <div class="section-title">Emerging Risks</div>
              {render_alert_rows(table_data)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="insight-card">
              <div class="section-title">Recent Risk Changes</div>
              {render_recent_rows(table_data)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_heatmap_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("<div class='section-title'>Impact by Likelihood</div>", unsafe_allow_html=True)
        render_materiality_legend()
        render_heatmap_chart(data, height="470px")

    with right:
        st.markdown("<div class='section-title'>Concentration</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-card">
              <div class="alert-row"><span>Top categories</span><b>{escape(format_counts(data['Category']))}</b></div>
              <div class="alert-row"><span>Top owners</span><b>{escape(format_counts(data['Owner']))}</b></div>
              <div class="alert-row"><span>High impact</span><b class="risk-high">{int(data['Impact'].isin(['High', 'Very High']).sum())}</b></div>
              <div class="alert-row"><span>High likelihood</span><b class="risk-med">{int(data['Likelihood'].isin(['High', 'Very High']).sum())}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-title'>Highest Ranked Risks</div>", unsafe_allow_html=True)
        render_risk_table(data, ["Risk", "Category", "Impact", "Likelihood", "Owner"])


def render_qoq_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    counts = data["Change"].value_counts().reindex(CHANGE_ORDER, fill_value=0)
    change_option = {
        "tooltip": {},
        "grid": {"top": 25, "right": 25, "bottom": 45, "left": 45},
        "xAxis": {
            "type": "category",
            "data": counts.index.tolist(),
            "axisLabel": {"color": "#9ca3af"},
        },
        "yAxis": {"type": "value", "axisLabel": {"color": "#9ca3af"}},
        "series": [{
            "type": "bar",
            "data": counts.values.tolist(),
            "barWidth": 24,
            "itemStyle": {"borderRadius": [8, 8, 0, 0], "color": "#f59e0b"},
            "label": {"show": True, "position": "top", "color": "#f5f5f5"},
        }],
    }

    chart_col, table_col = st.columns([0.9, 1.1])
    with chart_col:
        st.markdown("<div class='section-title'>Change Distribution</div>", unsafe_allow_html=True)
        st_echarts(change_option, height="360px")

    with table_col:
        st.markdown("<div class='section-title'>Movement Detail</div>", unsafe_allow_html=True)
        movement = data[data["Change"].isin(["New", "Increased", "Decreased", "Resolved"])]
        render_risk_table(movement, ["Risk", "Entity", "Category", "Impact", "Change", "Owner"])


def render_risk_library_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    st.markdown("<div class='section-title'>Inventory Records</div>", unsafe_allow_html=True)
    render_risk_table(data)


def render_controls_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    left, right = st.columns([1.25, 0.75])

    with left:
        st.markdown("<div class='section-title'>Control Review Queue</div>", unsafe_allow_html=True)
        render_risk_table(data, ["Risk", "Entity", "Impact", "Likelihood", "Control Focus", "Owner", "DQ Score"])

    with right:
        st.markdown("<div class='section-title'>Owner Coverage</div>", unsafe_allow_html=True)
        if data.empty:
            st.markdown("<div class='empty-state'>No controls in current slicer.</div>", unsafe_allow_html=True)
        else:
            owner_summary = (
                data.groupby("Owner")
                .agg(Risks=("Risk", "count"), Avg_DQ=("DQ Score", "mean"))
                .reset_index()
                .rename(columns={"Avg_DQ": "Avg DQ"})
                .sort_values(["Risks", "Avg DQ"], ascending=[False, True])
            )
            owner_summary["Avg DQ"] = owner_summary["Avg DQ"].round(0).astype(int)
            st.dataframe(owner_summary, width="stretch", hide_index=True)


def render_issues_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    issues = data[
        (data["DQ Score"] < 90)
        | (data.apply(lambda row: is_material_cell(row["Impact"], row["Likelihood"]), axis=1))
        | (data["Impact"].isin(["High", "Very High"]) & data["Change"].isin(["New", "Increased"]))
    ].copy()

    if not issues.empty:
        issues["Issue Type"] = issues.apply(
            lambda row: "DQ Exception" if row["DQ Score"] < 90 else "Material Movement",
            axis=1,
        )

    st.markdown("<div class='section-title'>Open Issues</div>", unsafe_allow_html=True)
    render_risk_table(issues, ["Risk", "Entity", "Issue Type", "Impact", "Likelihood", "Change", "Owner", "DQ Score"])


def render_ai_page(data: pd.DataFrame, filter_note: str) -> None:
    left, right = st.columns([1.1, 0.9])
    with left:
        render_chat_panel(data, filter_note, "page")
    with right:
        st.markdown("<div class='section-title'>Analyst Context</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-card">
              <p>{escape(build_ai_reply('Summarize this risk slice', data))}</p>
              <div class="tag amber">Filtered Context</div>
            </div>
            <div class="insight-card">
              <div class="section-title">Recommended Queue</div>
              {render_alert_rows(data)}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scenario_page(data: pd.DataFrame, filter_note: str) -> None:
    render_metric_cards(data, filter_note)
    scenario = st.selectbox(
        "Scenario",
        [
            "Critical vendor outage",
            "AI regulatory exam",
            "Cloud cost shock",
            "Fraud spike",
        ],
    )
    scenario_focus = {
        "Critical vendor outage": ["Technology", "Operational"],
        "AI regulatory exam": ["Compliance", "Regulatory"],
        "Cloud cost shock": ["Financial", "Technology"],
        "Fraud spike": ["Fraud", "Operational"],
    }
    focused = data[data["Category"].isin(scenario_focus[scenario])]
    if focused.empty:
        focused = data

    st.markdown(
        f"""
        <div class="insight-card">
          <div class="section-title">{escape(scenario)}</div>
          <p>
            Scenario lens applied to {len(focused)} relevant risks. Use this view to
            review owners, control focus, and records with high impact or weak DQ evidence.
          </p>
          <div class="tag amber">Scenario Lens</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_risk_table(focused, ["Risk", "Entity", "Category", "Impact", "Likelihood", "Control Focus", "Owner"])


def render_selected_page(page: str, data: pd.DataFrame, filter_note: str) -> None:
    if page == "Home":
        render_home_page(data, filter_note)
    elif page == "Risk Heatmap":
        render_heatmap_page(data, filter_note)
    elif page == "QoQ Changes":
        render_qoq_page(data, filter_note)
    elif page == "Risk Library":
        render_risk_library_page(data, filter_note)
    elif page == "Controls":
        render_controls_page(data, filter_note)
    elif page == "Issues":
        render_issues_page(data, filter_note)
    elif page == "AI Risk Analyst":
        render_ai_page(data, filter_note)
    elif page == "Scenario Analysis":
        render_scenario_page(data, filter_note)


if "risk_chat_messages" not in st.session_state:
    st.session_state.risk_chat_messages = [{
        "role": "assistant",
        "content": (
            "Ask me about the current slicer context. I can summarize exposure, "
            "rank material risks, explain quarter-over-quarter movement, or build a "
            "control review queue."
        ),
    }]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

if "home_heatmap_cell" not in st.session_state:
    st.session_state.home_heatmap_cell = None

if "home_heatmap_revision" not in st.session_state:
    st.session_state.home_heatmap_revision = 0

if "home_ring_filters" not in st.session_state:
    st.session_state.home_ring_filters = {"Category": None, "Owner Type": None}

if "home_ring_revision" not in st.session_state:
    st.session_state.home_ring_revision = 0


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div class='brand'>RISK INVEX</div>", unsafe_allow_html=True)
    st.markdown("<div class='go-button'>&lt; GO &gt;</div>", unsafe_allow_html=True)

    for section, pages in NAV_SECTIONS.items():
        st.markdown(f"### {section}")
        for page in pages:
            if st.session_state.selected_page == page:
                st.markdown(f"<div class='nav-active'>{escape(page)}</div>", unsafe_allow_html=True)
            elif st.button(page, key=f"nav_{page}", width="stretch"):
                st.session_state.selected_page = page
                st.rerun()

selected_page = st.session_state.selected_page


# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="top-header">
  <div>
    <div class="eyebrow">Enterprise Risk Command Center</div>
    <div class="title">{escape(PAGE_TITLES[selected_page])}</div>
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


render_selected_page(selected_page, filtered_data, filter_note)
