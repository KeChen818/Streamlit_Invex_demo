"""Streamlit view functions for dashboard tabs and controls."""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from .ai import ask_inventory_chatbot, get_compare_ai_analysis, get_taxonomy_ai_summary
from .settings import RISK_TABLE_COLUMNS
from .utils import unique_join


def render_ai_card(title: str, markdown_body: str) -> None:
    """Render a bordered narrative analysis block."""
    with st.container(border=True):
        st.markdown(
            f'<div class="ai-card-title">{escape(title)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(markdown_body)


def metric_cards(data: pd.DataFrame) -> None:
    """Render top-level count metrics for the filtered inventory."""
    material_count = int(data["Is_Material"].sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risks", f"{len(data):,}")
    col2.metric("Taxonomy L1 Groups", f"{data['Taxonomy_L1'].nunique():,}")
    col3.metric("Assessment Methods", f"{data['Assessment_Method'].nunique():,}")
    col4.metric("Risk Metrics", f"{data['Risk_Metric'].nunique():,}")
    col5.metric("Material Risks", f"{material_count:,}")


def render_group_summary(data: pd.DataFrame, group_summary: pd.DataFrame, model: str) -> None:
    """Render Taxonomy L1 charts, GPT summary, and risk popover cards."""
    left, right = st.columns([0.48, 0.52])

    with left:
        chart = (
            alt.Chart(group_summary)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("Risk_Count:Q", title="Risk count", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Taxonomy_L1:N", title=None, sort="-x"),
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
            .properties(height=max(260, 34 * len(group_summary)))
        )
        st.altair_chart(chart, width="stretch")

    with right:
        display = group_summary[
            [
                "Taxonomy_L1",
                "Risk_Count",
                "Material_Risks",
                "L2_Count",
                "Metrics",
                "Methods",
            ]
        ].rename(
            columns={
                "Taxonomy_L1": "Taxonomy L1",
                "Risk_Count": "Risks",
                "Material_Risks": "Material",
                "L2_Count": "Taxonomy L2",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)

    with st.spinner("Generating Taxonomy L1 analysis..."):
        summary_text = get_taxonomy_ai_summary(data, group_summary, model)
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
        analysis_text = get_compare_ai_analysis(scoped_data, model)
    render_ai_card("AI Method and Metric Analysis", analysis_text)


def render_risk_table(data: pd.DataFrame) -> None:
    """Render the associated-risk table used by the chatbot."""
    available_columns = [column for column in RISK_TABLE_COLUMNS if column in data.columns]
    register = data[available_columns].sort_values(
        ["Taxonomy_L1", "Risk_Title"],
        ascending=[True, True],
    )
    st.dataframe(
        register.rename(
            columns={
                "Group_ID": "ID",
                "Taxonomy_L0": "Taxonomy L0",
                "Taxonomy_L1": "Taxonomy L1",
                "Taxonomy_L2": "Taxonomy L2",
                "Risk_Title": "Risk",
                "Risk_Status": "Status",
                "Risk_Type": "Type",
                "Business_Division": "Business Division",
                "Risk_Metric": "Metric",
                "Assessment_Method": "Method",
                "Overall_Materiality": "Materiality",
                "Likelihood_Rating": "Likelihood Rating",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_ai_chatbot(data: pd.DataFrame, model: str) -> None:
    """Render the inventory chatbot and keep the table synced to returned IDs."""
    st.markdown(
        "Ask questions about the currently filtered inventory. The table below follows the risks selected by the answer."
    )

    if "risk_chat_messages" not in st.session_state:
        st.session_state.risk_chat_messages = []
    if "risk_chat_selected_ids" not in st.session_state:
        st.session_state.risk_chat_selected_ids = data["Group_ID"].head(25).tolist()

    if st.button("Clear chat"):
        st.session_state.risk_chat_messages = []
        st.session_state.risk_chat_selected_ids = data["Group_ID"].head(25).tolist()
        st.rerun()

    for message in st.session_state.risk_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about groups, divisions, GCRS, methods, metrics, or assumptions")
    if prompt:
        st.session_state.risk_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, ids = ask_inventory_chatbot(data, prompt, model)
            st.markdown(answer)

        st.session_state.risk_chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.risk_chat_selected_ids = ids

    valid_ids = set(data["Group_ID"])
    selected_ids = [group_id for group_id in st.session_state.risk_chat_selected_ids if group_id in valid_ids]
    associated = data[data["Group_ID"].isin(selected_ids)] if selected_ids else data
    st.subheader(f"Associated Risks ({len(associated)})")
    render_risk_table(associated)
