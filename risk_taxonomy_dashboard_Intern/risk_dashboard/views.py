"""Streamlit view functions for dashboard tabs and controls."""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from .ai import (
    ask_inventory_chatbot,
    get_compare_ai_analysis,
    get_taxonomy_ai_summary,
    get_theme_ai_analysis,
)
from .settings import CHAT_ASSOCIATED_RISK_LIMIT, RISK_TABLE_COLUMNS
from .themes import THEME_REVIEW_ACTIONS, build_theme_classification
from .utils import unique_join


THEME_DETAIL_RISK_TABLE_COLUMNS = [
    "Group_ID",
    "Taxonomy_L0",
    "Taxonomy_L1",
    "Taxonomy_L2",
    "Risk_Title",
    "Impact_Rating",
    "Impact_Numbers",
    "Business_Division",
    "Risk_Metric",
    "Assessment_Method",
    "Overall_Materiality",
    "Likelihood_Rating",
]

RISK_TABLE_COLUMN_LABELS = {
    "Group_ID": "ID",
    "Taxonomy_L0": "Taxonomy L0",
    "Taxonomy_L1": "Taxonomy L1",
    "Taxonomy_L2": "Taxonomy L2",
    "Risk_Title": "Risk",
    "Risk_Status": "Status",
    "Risk_Type": "Type",
    "Impact_Rating": "Impact Rating",
    "Impact_Numbers": "Impact Numbers",
    "Business_Division": "Business Division",
    "Risk_Metric": "Metric",
    "Assessment_Method": "Method",
    "Overall_Materiality": "Materiality",
    "Likelihood_Rating": "Likelihood Rating",
}


def render_ai_card(title: str, markdown_body: str) -> None:
    """Render a bordered narrative analysis block."""
    with st.container(border=True):
        st.markdown(
            f'<div class="ai-card-title">{escape(title)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(markdown_body)


def render_ai_error(title: str, error: Exception) -> None:
    """Render a clear AI setup or request failure without substituting local analysis."""
    with st.container(border=True):
        st.markdown(
            f'<div class="ai-card-title">{escape(title)}</div>',
            unsafe_allow_html=True,
        )
        st.error(f"{title} could not be generated. {error}")


def table_column_class(column: object) -> str:
    """Create a stable CSS class name for a rendered table column."""
    clean = "".join(character if character.isalnum() else "-" for character in str(column).lower())
    clean = "-".join(part for part in clean.split("-") if part)
    return f"col-{clean or 'value'}"


def format_compact_number(value: object) -> str:
    """Format numeric impact values as 0.00 / k / m / bn."""
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return str(value)

    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}bn"
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}m"
    if absolute >= 1_000:
        return f"{number / 1_000:.2f}k"
    return f"{number:.2f}"


def render_table_cell(value: object, column: str, bar_columns: set[str], bool_columns: set[str]) -> str:
    """Render a table cell with optional score-bar and boolean-badge treatments."""
    if column in bool_columns:
        truthy = bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"true", "yes", "1"}
        label = "True" if truthy else "False"
        badge_class = "true" if truthy else "false"
        return f'<span class="bool-badge bool-badge-{badge_class}">{label}</span>'

    if column in bar_columns:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return escape(str(value or ""))
        percent = max(0.0, min(1.0, score)) * 100
        return (
            '<div class="score-bar-cell">'
            '<div class="score-bar-track">'
            f'<span class="score-bar-fill" style="width: {percent:.0f}%"></span>'
            "</div>"
            f'<span class="score-bar-label">{score:.2f}</span>'
            "</div>"
        )

    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return ""
    if column == "Impact Numbers":
        return escape(format_compact_number(value))
    return escape(str(value))


def render_html_table(
    data: pd.DataFrame,
    max_rows: int | None = None,
    wide: bool = False,
    bar_columns: set[str] | None = None,
    bool_columns: set[str] | None = None,
) -> None:
    """Render a styled read-only table with a white body and grey header."""
    table_data = data.head(max_rows).copy() if max_rows else data.copy()
    wrapper_class = "risk-table-wrap risk-table-wrap-wide" if wide else "risk-table-wrap"
    table_class = "risk-html-table risk-html-table-wide" if wide else "risk-html-table"
    active_bar_columns = bar_columns or set()
    active_bool_columns = bool_columns or set()

    headers = "".join(
        f'<th class="{table_column_class(column)}">{escape(str(column))}</th>' for column in table_data.columns
    )
    body_rows = []
    for row in table_data.to_dict("records"):
        cells = "".join(
            f'<td class="{table_column_class(column)}">'
            f"{render_table_cell(row.get(column, ''), str(column), active_bar_columns, active_bool_columns)}"
            "</td>"
            for column in table_data.columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    html = f'<table class="{table_class}"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    st.markdown(f'<div class="{wrapper_class}">{html}</div>', unsafe_allow_html=True)


def metric_cards(data: pd.DataFrame) -> None:
    """Render top-level count metrics for the filtered inventory."""
    material_count = int(data["Is_Material"].sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Risks", f"{len(data):,}")
    col2.metric("Taxonomy L1", f"{data['Taxonomy_L1'].nunique():,}")
    col3.metric("Assessment Methods", f"{data['Assessment_Method'].nunique():,}")
    col4.metric("Risk Metrics", f"{data['Risk_Metric'].nunique():,}")
    col5.metric("Material Risks", f"{material_count:,}")


def render_group_summary(data: pd.DataFrame, group_summary: pd.DataFrame, model: str) -> None:
    """Render Taxonomy L1 charts, GPT summary, and risk popover cards."""
    chart_summary = group_summary.head(15).copy()
    max_risk_count = float(chart_summary["Risk_Count"].max()) if not chart_summary.empty else 1.0
    x_domain_max = max(1.0, max_risk_count * 1.18)
    left, right = st.columns([0.48, 0.52])

    with left:
        base = (
            alt.Chart(chart_summary)
            .encode(
                x=alt.X(
                    "Risk_Count:Q",
                    title="Risk count",
                    axis=alt.Axis(tickMinStep=1),
                    scale=alt.Scale(domain=[0, x_domain_max]),
                ),
                y=alt.Y("Taxonomy_L1:N", title=None, sort="-x"),
            )
        )
        bars = (
            base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                color=alt.Color(
                    "Risk_Count:Q",
                    scale=alt.Scale(range=["#f0c9c4", "#d64545"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Taxonomy_L1:N", title="Group"),
                    alt.Tooltip("Risk_Count:Q", title="Risks"),
                    alt.Tooltip("Material_Risks:Q", title="Material risks"),
                    alt.Tooltip("L2_Count:Q", title="Taxonomy L2"),
                ],
            )
        )
        labels = base.mark_text(
            align="left",
            baseline="middle",
            color="#3a3630",
            dx=6,
            fontSize=12,
            fontWeight="bold",
        ).encode(text=alt.Text("Risk_Count:Q", format=",d"))
        chart = (
            (bars + labels)
            .properties(height=max(260, 34 * len(chart_summary)))
        )
        st.altair_chart(chart, width="stretch")
        st.caption(f"Showing top {len(chart_summary):,} Taxonomy L1 groups by risk count.")

    with right:
        display = group_summary[
            [
                "Taxonomy_L1",
                "Risk_Count",
                "Associated_SubLegal_Entity",
                "Associated_Business_Division",
                "Metrics",
                "Methods",
            ]
        ].rename(
            columns={
                "Taxonomy_L1": "Taxonomy L1",
                "Risk_Count": "Risks",
                "Associated_SubLegal_Entity": "Associated SubLegal Entity",
                "Associated_Business_Division": "Associated Business Division",
            }
        )
        render_html_table(display)

    with st.spinner("Generating Taxonomy L1 analysis..."):
        try:
            summary_text = get_taxonomy_ai_summary(data, group_summary, model)
        except Exception as exc:
            render_ai_error("AI Taxonomy Summary", exc)
        else:
            render_ai_card("AI Taxonomy Summary", summary_text)

    st.subheader("Risks Under Each Taxonomy L1 Group")
    card_columns = st.columns(3)
    for idx, row in enumerate(group_summary.to_dict("records")):
        group_name = row["Taxonomy_L1"]
        group_risks = data[data["Taxonomy_L1"] == group_name].sort_values("Risk_Title")
        group_metrics = unique_join(group_risks["Risk_Metric"], limit=3)
        group_methods = unique_join(group_risks["Assessment_Method"], limit=2)

        with card_columns[idx % 3]:
            st.markdown(
                f"""
                <div class="taxonomy-card">
                    <h3>{escape(group_name)}</h3>
                    <div class="taxonomy-meta">{int(row["Risk_Count"])} risks | {int(row["L2_Count"])} taxonomy L2</div>
                    <div class="taxonomy-meta"><b>Metrics:</b> {escape(group_metrics)}</div>
                    <div class="taxonomy-meta"><b>Methods:</b> {escape(group_methods)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.popover(f"View {int(row['Risk_Count'])} risks"):
                for risk in group_risks.to_dict("records"):
                    st.markdown(
                        f"""
                        <div class="risk-pop-card">
                            <b>{escape(risk["Risk_Title"])}</b>
                            <span>ID: {escape(risk["Group_ID"])} | Status: {escape(risk["Risk_Status"])}</span>
                            <span>Taxonomy L2: {escape(risk["Taxonomy_L2"])}</span>
                            <span>Metric: {escape(risk["Risk_Metric"])}</span>
                            <span>Method: {escape(risk["Assessment_Method"])}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_comparison(
    comparison: pd.DataFrame,
    group_summary: pd.DataFrame,
    data: pd.DataFrame,
    model: str,
) -> None:
    """Render method/metric ring charts and GPT comparison analysis."""
    group_options = group_summary["Taxonomy_L1"].tolist()
    default_groups = group_options[: min(3, len(group_options))]

    selected_groups = st.multiselect(
        "Taxonomy L1 groups",
        group_options,
        default=default_groups,
    )
    scoped = comparison[comparison["Taxonomy_L1"].isin(selected_groups)] if selected_groups else comparison
    scoped_data = data[data["Taxonomy_L1"].isin(selected_groups)] if selected_groups else data

    color_range = ["#d64545", "#1a1814", "#8f6b4a", "#5c5650", "#c09b70", "#7d746b"]
    method_counts = (
        scoped.groupby("Assessment_Method", as_index=False)["Risk_Count"]
        .sum()
        .sort_values("Risk_Count", ascending=False)
    )
    metric_counts = (
        scoped.groupby("Risk_Metric", as_index=False)["Risk_Count"]
        .sum()
        .sort_values("Risk_Count", ascending=False)
    )

    left, right = st.columns(2)
    with left:
        chart = (
            alt.Chart(method_counts)
            .mark_arc(innerRadius=72, outerRadius=118, cornerRadius=4, stroke="#f5f2ec", strokeWidth=2)
            .encode(
                theta=alt.Theta("Risk_Count:Q", title="Risks"),
                color=alt.Color(
                    "Assessment_Method:N",
                    title="Assessment method",
                    scale=alt.Scale(range=color_range),
                ),
                tooltip=[
                    alt.Tooltip("Assessment_Method:N", title="Method"),
                    alt.Tooltip("Risk_Count:Q", title="Risks"),
                ],
            )
            .properties(height=330, title="Assessment Method Mix")
        )
        st.altair_chart(chart, width="stretch")

    with right:
        chart = (
            alt.Chart(metric_counts)
            .mark_arc(innerRadius=72, outerRadius=118, cornerRadius=4, stroke="#f5f2ec", strokeWidth=2)
            .encode(
                theta=alt.Theta("Risk_Count:Q", title="Risks"),
                color=alt.Color(
                    "Risk_Metric:N",
                    title="Risk metric",
                    scale=alt.Scale(range=color_range),
                ),
                tooltip=[
                    alt.Tooltip("Risk_Metric:N", title="Metric"),
                    alt.Tooltip("Risk_Count:Q", title="Risks"),
                ],
            )
            .properties(height=330, title="Risk Metric Mix")
        )
        st.altair_chart(chart, width="stretch")

    with st.spinner("Generating method and metric analysis..."):
        try:
            analysis_text = get_compare_ai_analysis(scoped_data, model)
        except Exception as exc:
            render_ai_error("AI Method and Metric Analysis", exc)
        else:
            render_ai_card("AI Method and Metric Analysis", analysis_text)


def format_list_cell(values: object) -> str:
    """Format list-valued theme fields for display tables."""
    if isinstance(values, list):
        return ", ".join(str(value) for value in values)
    return str(values or "")


def wrap_chart_label(value: object, width: int = 22) -> str:
    """Split a chart axis label into two readable lines without truncating it."""
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= width:
        return text

    words = text.split()
    if len(words) <= 1:
        return text

    best_split = 1
    best_score = float("inf")
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        score = abs(len(left) - len(right)) + max(0, len(left) - width) + max(0, len(right) - width)
        if score < best_score:
            best_score = score
            best_split = index

    return f"{' '.join(words[:best_split])}\n{' '.join(words[best_split:])}"


def sync_theme_review_state(themes: pd.DataFrame) -> None:
    """Persist review-control values into the shared review state."""
    if "theme_review_state" not in st.session_state:
        st.session_state.theme_review_state = {}

    for theme in themes.to_dict("records"):
        theme_id = theme["Theme_ID"]
        name_key = f"review_name_{theme_id}"
        action_key = f"review_action_{theme_id}"
        reviewer_key = f"reviewer_{theme_id}"
        notes_key = f"review_notes_{theme_id}"
        if name_key in st.session_state:
            st.session_state.theme_review_state[theme_id] = {
                "Suggested Risk Theme": st.session_state[name_key],
                "Review Action": st.session_state.get(action_key, theme["Review_Action"]),
                "Reviewed By": st.session_state.get(reviewer_key, ""),
                "Human Notes": st.session_state.get(notes_key, ""),
            }


def render_theme_review_editor(themes: pd.DataFrame) -> None:
    """Render the human review workflow table."""
    if themes.empty:
        st.info("No suggested themes met the 0.50 confidence threshold for review.")
        return

    sync_theme_review_state(themes)

    rows = []
    for theme in themes.to_dict("records"):
        state = st.session_state.theme_review_state.get(theme["Theme_ID"], {})
        rows.append(
            {
                "Theme ID": theme["Theme_ID"],
                "Suggested Risk Theme": state.get("Suggested Risk Theme", theme["Theme_Name"]),
                "Risk Count": theme["Risk_Count"],
                "Business Divisions": format_list_cell(theme["Business_Divisions"]),
                "Taxonomy Alignment": format_list_cell(theme["Taxonomy_Alignment"]),
                "Common Topic": theme.get("Common_Topic", ""),
                "Key Drivers": format_list_cell(theme.get("Key_Drivers", [])),
                "Confidence Score": theme.get("Confidence_Score", 0.0),
                "Review Required": theme.get("Review_Required", False),
                "Governance Check": theme.get("Governance_Check", ""),
                "Risk IDs": format_list_cell(theme["Risk_IDs"]),
                "Potential Gap": theme["Potential_Gap"],
                "Review Action": state.get("Review Action", theme["Review_Action"]),
                "Reviewed By": state.get("Reviewed By", ""),
                "Human Notes": state.get("Human Notes", ""),
            }
        )

    render_html_table(
        pd.DataFrame(rows),
        wide=True,
        bar_columns={"Confidence Score"},
        bool_columns={"Review Required"},
    )


def render_theme_review_controls(themes: pd.DataFrame) -> None:
    """Render collapsed reviewer inputs for updating theme decisions."""
    if themes.empty:
        st.info("No suggested themes met the 0.50 confidence threshold for review.")
        return

    sync_theme_review_state(themes)
    st.markdown('<div class="review-controls-title">Update review decisions</div>', unsafe_allow_html=True)
    for theme in themes.to_dict("records"):
        theme_id = theme["Theme_ID"]
        state = st.session_state.theme_review_state.get(theme_id, {})
        name_key = f"review_name_{theme_id}"
        action_key = f"review_action_{theme_id}"
        reviewer_key = f"reviewer_{theme_id}"
        notes_key = f"review_notes_{theme_id}"

        st.session_state.setdefault(name_key, state.get("Suggested Risk Theme", theme["Theme_Name"]))
        st.session_state.setdefault(action_key, state.get("Review Action", theme["Review_Action"]))
        st.session_state.setdefault(reviewer_key, state.get("Reviewed By", ""))
        st.session_state.setdefault(notes_key, state.get("Human Notes", ""))

        current_action = st.session_state[action_key]
        if current_action not in THEME_REVIEW_ACTIONS:
            st.session_state[action_key] = theme["Review_Action"]

        with st.container(border=True):
            st.markdown(f"**{escape(theme_id)} | {escape(theme['Theme_Name'])}**")
            name_col, action_col, reviewer_col = st.columns([2.3, 1.2, 1.1])
            name_col.text_input("Suggested Risk Theme", key=name_key)
            action_col.selectbox(
                "Review Action",
                options=THEME_REVIEW_ACTIONS,
                index=THEME_REVIEW_ACTIONS.index(st.session_state[action_key]),
                key=action_key,
            )
            reviewer_col.text_input("Reviewed By", key=reviewer_key)
            st.text_area("Human Notes", key=notes_key, height=72)


def render_theme_detail_cards(data: pd.DataFrame, themes: pd.DataFrame) -> None:
    """Render expandable cards with theme summaries and mapped risks."""
    st.subheader("Theme Detail")
    st.caption("Open a theme to review the AI-generated summary, core metrics, and the underlying risk records.")
    if themes.empty:
        st.info("No theme details are available because no suggested themes met the 0.50 confidence threshold.")
        return

    for theme in themes.sort_values(["Risk_Count", "Average_Similarity"], ascending=[False, False]).to_dict(
        "records"
    ):
        risk_ids = set(theme["Risk_IDs"])
        theme_risks = data[data["Group_ID"].astype(str).isin(risk_ids)]
        label = f"{theme['Theme_ID']} | {theme['Theme_Name']} ({int(theme['Risk_Count'])} risks)"
        with st.expander(label):
            st.markdown(theme["Theme_Summary"])
            cols = st.columns(4)
            cols[0].metric("Risks", f"{int(theme['Risk_Count']):,}")
            cols[1].metric("Material Risks", f"{int(theme.get('Material_Risks', 0)):,}")
            cols[2].metric("Divisions", f"{len(theme['Business_Divisions'])}")
            cols[3].metric("Metrics", f"{theme_risks['Risk_Metric'].nunique()}")
            render_risk_table(theme_risks, columns=THEME_DETAIL_RISK_TABLE_COLUMNS)


def render_count_ring_chart(summary: pd.DataFrame, dimension: str, title: str, chart_height: int = 220) -> None:
    """Render a compact ring chart for selected-scope counts."""
    chart_data = summary[summary["Dimension"] == dimension].copy()
    if chart_data.empty:
        st.info(f"No {dimension} data is available for the current filter.")
        return

    chart_data = chart_data.sort_values("Value").copy()
    total_count = max(1, int(chart_data["Risk_Count"].sum()))
    chart_data["Share"] = chart_data["Risk_Count"] / total_count
    chart_data["Count_Label"] = chart_data["Risk_Count"].where(chart_data["Share"] >= 0.08, "")

    arcs = (
        alt.Chart(chart_data)
        .mark_arc(innerRadius=42, outerRadius=72, stroke="#fff", strokeWidth=2)
        .encode(
            theta=alt.Theta("Risk_Count:Q", title="Risk count", stack=True),
            order=alt.Order("Value:N", sort="ascending"),
            color=alt.Color(
                "Value:N",
                title=dimension,
                scale=alt.Scale(range=["#d64545", "#1a1814", "#8f6b4a", "#5c5650", "#c09b70", "#7d746b"]),
            ),
            tooltip=[
                alt.Tooltip("Value:N", title=dimension),
                alt.Tooltip("Risk_Count:Q", title="Risks"),
            ],
        )
    )
    labels = (
        alt.Chart(chart_data)
        .mark_text(
            radius=57,
            color="#fff",
            fontSize=10,
            fontWeight="bold",
            baseline="middle",
            align="center",
        )
        .encode(
            theta=alt.Theta("Risk_Count:Q", stack="center"),
            order=alt.Order("Value:N", sort="ascending"),
            text=alt.Text("Count_Label:N"),
        )
    )
    chart = (
        (arcs + labels)
        .properties(
            height=chart_height,
            title=alt.TitleParams(text=title, anchor="middle", offset=10),
            padding={"top": 24, "right": 18, "bottom": 36, "left": 18},
        )
    )
    st.altair_chart(chart, width="stretch")


def render_theme_classification(data: pd.DataFrame, model: str, show_heading: bool = True) -> None:
    """Render AI-assisted risk theme classification and reviewer workflow."""
    if show_heading:
        st.subheader("AI Risk Theme Classification")

    record_count = len(data)
    default_target = min(record_count, max(5, min(30, round(record_count / 12))))
    if record_count >= 240:
        default_target = min(record_count, 25)

    target_theme_count = max(1, default_target)
    top_k = 8
    relationship_threshold = 0.72
    with st.expander("Theme grouping controls", expanded=False):
        st.markdown(
            f"""
            - **Target themes (max):** Sets the executive-level upper bound. The default is about one theme per 12 risks, capped at 30, and 25 for inventories above 240 risks.
            - **Similar risks per risk:** Controls how many nearest neighbors each risk checks before clustering. Higher values find broader relationships but can add noise.
            - **Relationship threshold:** Filters weak pairings using the hybrid score. `0.72` means related, while `0.82+` is treated as same-theme evidence.
            """
        )
        control_1, control_2, control_3 = st.columns(3)
        with control_1:
            target_theme_count = st.slider(
                "Target themes (max)",
                min_value=1,
                max_value=max(1, min(60, record_count)),
                value=target_theme_count,
                step=1,
            )
        with control_2:
            top_k = st.slider("Similar risks per risk", min_value=2, max_value=15, value=top_k, step=1)
        with control_3:
            relationship_threshold = st.slider(
                "Relationship threshold",
                min_value=0.50,
                max_value=0.95,
                value=relationship_threshold,
                step=0.01,
            )

    try:
        themes, relationships, dimension_summary = build_theme_classification(
            data,
            top_k=top_k,
            relationship_threshold=relationship_threshold,
            target_theme_count=target_theme_count,
        )
    except Exception as exc:
        st.error(f"AI Risk Theme Classification could not run. {exc}")
        return

    st.caption(
        f"{len(themes):,} suggested themes are shown. Themes with confidence below 0.50, fewer than 2 risks, "
        "or zero average internal similarity are excluded from the charts, AI summary, review workflow, and detail cards."
    )

    chart_height = 220
    chart_left, chart_right = st.columns([0.40, 0.60])
    with chart_left:
        if themes.empty:
            st.info("No suggested themes met the final theme criteria for the current filter.")
        else:
            chart_data = themes.sort_values("Risk_Count", ascending=False).head(10).copy()
            chart_data["Theme_Label"] = chart_data["Theme_Name"].apply(lambda value: wrap_chart_label(value))
            theme_sort = chart_data["Theme_Label"].tolist()
            max_count = max(1, int(chart_data["Risk_Count"].max()))
            x_domain = [0, max_count + max(1, round(max_count * 0.22))]
            materiality_data = chart_data.melt(
                id_vars=["Theme_ID", "Theme_Name", "Theme_Label", "Risk_Count", "Average_Similarity"],
                value_vars=["Material_Risks", "Non_Material_Risks"],
                var_name="Materiality",
                value_name="Materiality_Count",
            )
            materiality_data["Materiality"] = materiality_data["Materiality"].replace(
                {
                    "Material_Risks": "Material",
                    "Non_Material_Risks": "Non-Material",
                }
            )
            materiality_data["Materiality_Order"] = materiality_data["Materiality"].map(
                {"Material": 0, "Non-Material": 1}
            )
            materiality_data = materiality_data.sort_values(
                ["Theme_Label", "Materiality_Order"],
                ascending=[True, True],
            )
            materiality_data["Segment_End"] = materiality_data.groupby("Theme_Label")[
                "Materiality_Count"
            ].cumsum()
            materiality_data["Segment_Start"] = materiality_data["Segment_End"] - materiality_data[
                "Materiality_Count"
            ]
            materiality_data["Segment_Center"] = (
                materiality_data["Segment_Start"] + (materiality_data["Materiality_Count"] / 2)
            )
            materiality_label_data = materiality_data[materiality_data["Materiality_Count"] > 0].copy()
            bars = (
                alt.Chart(materiality_data)
                .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                .encode(
                    x=alt.X(
                        "Segment_Start:Q",
                        title="Risk count",
                        axis=alt.Axis(tickMinStep=1),
                        scale=alt.Scale(domain=x_domain),
                    ),
                    x2=alt.X2("Segment_End:Q"),
                    y=alt.Y(
                        "Theme_Label:N",
                        title=None,
                        sort=theme_sort,
                        axis=alt.Axis(labelLimit=220, labelLineHeight=13),
                    ),
                    color=alt.Color(
                        "Materiality:N",
                        title="Materiality",
                        scale=alt.Scale(domain=["Material", "Non-Material"], range=["#d64545", "#7d746b"]),
                    ),
                    tooltip=[
                        alt.Tooltip("Theme_ID:N", title="Theme ID"),
                        alt.Tooltip("Theme_Name:N", title="Theme"),
                        alt.Tooltip("Risk_Count:Q", title="Risks"),
                        alt.Tooltip("Materiality:N", title="Materiality"),
                        alt.Tooltip("Materiality_Count:Q", title="Materiality count"),
                        alt.Tooltip("Average_Similarity:Q", title="Avg similarity", format=".3f"),
                    ],
                )
            )
            labels = (
                alt.Chart(materiality_label_data)
                .mark_text(
                    align="center",
                    baseline="middle",
                    color="#fff",
                    stroke="#5c5650",
                    strokeWidth=0.25,
                    fontSize=11,
                    fontWeight="bold",
                )
                .encode(
                    x=alt.X("Segment_Center:Q", scale=alt.Scale(domain=x_domain)),
                    y=alt.Y("Theme_Label:N", title=None, sort=theme_sort),
                    text=alt.Text("Materiality_Count:Q", format="d"),
                )
            )
            total_labels = (
                alt.Chart(chart_data)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    color="#5c5650",
                    fontSize=10,
                    fontWeight="bold",
                )
                .encode(
                    x=alt.X("Risk_Count:Q", scale=alt.Scale(domain=x_domain)),
                    y=alt.Y("Theme_Label:N", title=None, sort=theme_sort),
                    text=alt.Text("Risk_Count:Q", format="d"),
                )
            )
            chart = (bars + labels + total_labels).properties(
                height=max(chart_height, 34 * len(chart_data)),
                title="Suggested Risk Themes by Overall_Materiality Count",
            )
            st.altair_chart(chart, width="stretch")

    with chart_right:
        if dimension_summary.empty:
            st.info("No Business Division or GCRS data is available for the current filter.")
        else:
            ring_left, ring_right = st.columns(2)
            with ring_left:
                render_count_ring_chart(
                    dimension_summary,
                    "Business Division",
                    "Business Division Count",
                    chart_height=chart_height,
                )
            with ring_right:
                render_count_ring_chart(
                    dimension_summary,
                    "GCRS",
                    "GCRS Count",
                    chart_height=chart_height,
                )

    st.subheader("AI Theme Analysis")
    with st.spinner("Generating theme analysis..."):
        try:
            theme_summary_text, theme_analysis_text = get_theme_ai_analysis(data, themes, relationships, model)
        except Exception as exc:
            theme_summary_text = ""
            theme_analysis_text = ""
            theme_error = exc
        else:
            theme_error = None
    summary_box, analysis_box = st.columns([0.42, 0.58])
    with summary_box:
        if theme_error:
            render_ai_error("Theme Summary", theme_error)
        else:
            render_ai_card("Theme Summary", theme_summary_text)
    with analysis_box:
        if theme_error:
            render_ai_error("Theme Analysis", theme_error)
        else:
            render_ai_card("Theme Analysis", theme_analysis_text)

    render_theme_detail_cards(data, themes)

    st.subheader("Human Review Workflow")
    st.caption(
        "Use this section to accept, rename, merge, split, reject, or annotate suggested themes before they become final."
    )
    with st.expander("Open human review workflow", expanded=False):
        render_theme_review_editor(themes)
    with st.expander("Update review decisions", expanded=False):
        render_theme_review_controls(themes)

    st.subheader("Similar Risk Relationships")
    st.caption(
        "Open the relationship table to inspect the risk-to-risk evidence behind each suggested theme and confidence score."
    )
    with st.expander("Open similar risk relationship table", expanded=False):
        if relationships.empty:
            st.info("No similar-risk relationships met the selected threshold.")
        else:
            display = relationships.rename(
                columns={
                    "Risk_ID": "Risk ID",
                    "Risk_Title": "Risk",
                    "Similar_Risk_ID": "Similar Risk ID",
                    "Similar_Risk_Title": "Similar Risk",
                    "Similarity_Score": "Score",
                    "Semantic_Similarity": "Title/Description",
                    "Metric_Impact_Similarity": "Metric/Impact",
                    "Taxonomy_Alignment_Score": "Taxonomy",
                    "Driver_Similarity": "Driver",
                    "Mixed_Category_Warning": "Mixed Category",
                    "Same_Taxonomy_L1": "Same Taxonomy L1",
                    "Same_Metric": "Same Metric",
                    "Same_Method": "Same Method",
                }
            )
            render_html_table(
                display,
                max_rows=120,
                wide=True,
                bar_columns={"Score", "Title/Description", "Metric/Impact", "Taxonomy", "Driver"},
                bool_columns={"Mixed Category", "Same Taxonomy L1", "Same Metric", "Same Method"},
            )


def render_risk_table(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> None:
    """Render the associated-risk table used by the chatbot."""
    source_columns = columns or RISK_TABLE_COLUMNS
    available_columns = [column for column in source_columns if column in data.columns]
    register = data[available_columns].sort_values(
        ["Taxonomy_L1", "Risk_Title"],
        ascending=[True, True],
    )
    render_html_table(register.rename(columns=RISK_TABLE_COLUMN_LABELS), max_rows=max_rows)


def render_ai_chatbot(data: pd.DataFrame, model: str) -> None:
    """Render the inventory chatbot and keep the table synced to returned IDs."""
    st.markdown(
        "Ask questions about the currently filtered inventory. The table below follows the risks selected by the answer."
    )

    if "risk_chat_messages" not in st.session_state:
        st.session_state.risk_chat_messages = []
    if "risk_chat_selected_ids" not in st.session_state:
        st.session_state.risk_chat_selected_ids = data["Group_ID"].astype(str).head(CHAT_ASSOCIATED_RISK_LIMIT).tolist()

    if st.button("Clear chat"):
        st.session_state.risk_chat_messages = []
        st.session_state.risk_chat_selected_ids = data["Group_ID"].astype(str).head(CHAT_ASSOCIATED_RISK_LIMIT).tolist()
        st.rerun()

    for message in st.session_state.risk_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about groups, divisions, GCRS, methods, metrics, or assumptions")
    if prompt:
        prior_messages = st.session_state.risk_chat_messages[-8:]
        prior_scope = st.session_state.risk_chat_selected_ids
        st.session_state.risk_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, ids = ask_inventory_chatbot(
                        data,
                        prompt,
                        model,
                        chat_history=prior_messages,
                        prior_group_ids=prior_scope,
                        max_ids=CHAT_ASSOCIATED_RISK_LIMIT,
                    )
                except Exception as exc:
                    answer = f"AI chat could not answer. {exc}"
                    ids = []
                    st.error(answer)
                else:
                    st.markdown(answer)

        st.session_state.risk_chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "matching_group_ids": ids,
            }
        )
        st.session_state.risk_chat_selected_ids = ids

    valid_ids = set(data["Group_ID"].astype(str))
    selected_ids = [group_id for group_id in st.session_state.risk_chat_selected_ids if group_id in valid_ids]
    associated = data[data["Group_ID"].astype(str).isin(selected_ids)]
    st.subheader(f"Associated Risks ({len(associated)})")
    st.caption(
        f"Table follows the latest discussed Group ID scope from the chat. "
        f"Up to {CHAT_ASSOCIATED_RISK_LIMIT:,} associated risks are shown from the current sidebar filters."
    )
    render_risk_table(associated, max_rows=CHAT_ASSOCIATED_RISK_LIMIT)
