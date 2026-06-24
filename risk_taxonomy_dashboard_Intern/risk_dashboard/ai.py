"""OpenAI integration and chatbot logic."""

import json
import os
import re
from typing import Any

import pandas as pd
import streamlit as st
from openai import OpenAI

from .settings import AI_CONTEXT_COLUMNS, CHAT_ASSOCIATED_RISK_LIMIT, DEFAULT_MODEL
from .utils import format_top_counts, top_counts


def selected_model() -> str:
    """Resolve the backend model without exposing it in the UI."""
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def require_openai_api_key() -> str:
    """Return the configured OpenAI API key or raise a clear setup error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for AI analysis in the intern dashboard.")
    return api_key


def records_json(data: pd.DataFrame, limit: int | None = None) -> str:
    """Serialize dataframe records to JSON using pandas-safe scalar conversion."""
    frame = data.head(limit) if limit else data
    return frame.to_json(orient="records", force_ascii=False, indent=2)


def build_inventory_context(data: pd.DataFrame, limit: int = 80) -> str:
    """Serialize filtered inventory records for GPT prompts."""
    columns = [column for column in AI_CONTEXT_COLUMNS if column in data.columns]
    return records_json(data[columns], limit=limit)


def count_records(data: pd.DataFrame, columns: list[str], limit: int = 20) -> list[dict[str, Any]]:
    """Build compact count records for prompt grounding."""
    summary = (
        data.groupby(columns, dropna=False)
        .agg(Risk_Count=("Group_ID", "nunique"))
        .reset_index()
        .sort_values("Risk_Count", ascending=False)
        .head(limit)
    )
    return json.loads(summary.to_json(orient="records", force_ascii=False))


def build_taxonomy_prompt_facts(data: pd.DataFrame, group_summary: pd.DataFrame) -> str:
    """Build computed facts that ground the Taxonomy L1 GPT prompt."""
    group_columns = [
        "Taxonomy_L1",
        "Risk_Count",
        "Material_Risks",
        "Associated_Business_Division",
        "Metrics",
        "Methods",
        "L2_Count",
    ]
    available_group_columns = [column for column in group_columns if column in group_summary.columns]
    facts = {
        "total_risks": int(len(data)),
        "taxonomy_l1_count": int(data["Taxonomy_L1"].nunique()),
        "largest_taxonomy_l1_groups": format_top_counts(top_counts(data["Taxonomy_L1"], limit=5)),
        "taxonomy_l1_summary": json.loads(
            group_summary[available_group_columns].head(20).to_json(orient="records", force_ascii=False)
        ),
        "business_division_taxonomy_counts": count_records(
            data,
            ["Business_Division", "Taxonomy_L1"],
            limit=30,
        ),
        "business_division_metric_counts": count_records(
            data,
            ["Business_Division", "Risk_Metric"],
            limit=30,
        ),
        "business_division_method_counts": count_records(
            data,
            ["Business_Division", "Assessment_Method"],
            limit=30,
        ),
    }
    return json.dumps(facts, ensure_ascii=False, indent=2)


def build_compare_prompt_facts(data: pd.DataFrame) -> str:
    """Build computed facts that ground the method and metric GPT prompt."""
    comparable = data[
        [
            "Taxonomy_L1",
            "Assessment_Method",
            "Risk_Metric",
            "Business_Division",
            "GCRS",
            "Group_ID",
            "Impact_Comment",
        ]
    ].copy()
    facts = {
        "total_risks": int(len(data)),
        "taxonomy_method_metric_counts": count_records(
            data,
            ["Taxonomy_L1", "Assessment_Method", "Risk_Metric"],
            limit=40,
        ),
        "taxonomy_division_method_counts": count_records(
            data,
            ["Taxonomy_L1", "Business_Division", "Assessment_Method"],
            limit=40,
        ),
        "taxonomy_division_metric_counts": count_records(
            data,
            ["Taxonomy_L1", "Business_Division", "Risk_Metric"],
            limit=40,
        ),
        "comparable_risk_records": json.loads(
            comparable.head(80).to_json(orient="records", force_ascii=False)
        ),
    }
    return json.dumps(facts, ensure_ascii=False, indent=2)


def format_chat_history(messages: list[dict[str, str]] | None, limit: int = 8) -> str:
    """Condense recent chatbot messages for follow-up question context."""
    if not messages:
        return "No previous messages in this chat."

    lines = []
    for message in messages[-limit:]:
        role = str(message.get("role", "message")).strip().title()
        content = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
        if len(content) > 900:
            content = content[:900].rsplit(" ", 1)[0] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No previous messages in this chat."


@st.cache_data(show_spinner=False, ttl=900)
def request_openai_text(model: str, system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
    """Call the OpenAI Responses API and cache repeated analysis requests."""
    client = OpenAI(api_key=require_openai_api_key())
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def get_taxonomy_ai_summary(data: pd.DataFrame, group_summary: pd.DataFrame, model: str) -> str:
    """Generate the Taxonomy L1 narrative with GPT."""
    system_prompt = (
        "You are a risk inventory analyst. Use only the supplied risk records and computed facts. "
        "Do not invent records, counts, divisions, metrics, methods, or assumptions. "
        "Do not analyze, aggregate, or compare Impact_Numbers. "
        "Write concise Markdown."
    )
    user_prompt = f"""
Analyze this filtered risk inventory for the Taxonomy L1 page.

Answer:
1. Summarize how many Taxonomy L1 risk groups exist and which groups are largest by risk count.
2. For each Business_Division, identify the dominant Taxonomy_L1 groups and dominant Risk_Metric values used.
3. Identify potential Risk_Metric or Assessment_Method gaps for Taxonomy_L1 groups across Business_Division.

Computed facts:
{build_taxonomy_prompt_facts(data, group_summary)}

Risk records:
{build_inventory_context(data)}
"""
    return request_openai_text(model, system_prompt, user_prompt, 1800)


def get_compare_ai_analysis(data: pd.DataFrame, model: str) -> str:
    """Generate the method/metric narrative with GPT."""
    system_prompt = (
        "You are a risk inventory analyst. Use only the supplied risk records and computed facts. "
        "Do not invent records. Do not analyze Impact_Numbers. "
        "You may compare Impact_Comment text only to explain assumption differences. "
        "Write concise Markdown."
    )
    user_prompt = f"""
Analyze this filtered risk inventory for the Method and Metric Compare page.

Keep the response in two sections:
1. How similar or different each Taxonomy_L1 group is by comparing Assessment_Method and Risk_Metric across Business_Division, GCRS, and Group_ID.
2. For risks with the same Taxonomy_L1 group, same Assessment_Method, and similar or same Risk_Metric, identify assumption differences in Impact_Comment.

Computed facts:
{build_compare_prompt_facts(data)}

Risk records:
{build_inventory_context(data)}
"""
    return request_openai_text(model, system_prompt, user_prompt, 2000)


def split_theme_ai_sections(text: str) -> tuple[str, str]:
    """Split a two-section GPT theme response into dashboard card bodies."""
    clean = text.strip()
    if not clean:
        raise ValueError("OpenAI returned an empty theme analysis.")

    summary_match = re.search(r"(?im)^#{1,3}\s*Theme Summary\s*$", clean)
    analysis_match = re.search(r"(?im)^#{1,3}\s*Theme Analysis\s*$", clean)
    if not summary_match or not analysis_match or analysis_match.start() <= summary_match.end():
        raise ValueError("Theme analysis must include '## Theme Summary' and '## Theme Analysis' headings.")

    summary = clean[summary_match.end() : analysis_match.start()].strip()
    analysis = clean[analysis_match.end() :].strip()
    if not summary or not analysis:
        raise ValueError("Theme analysis response is missing one of the required sections.")
    return summary, analysis


def get_theme_ai_analysis(
    data: pd.DataFrame,
    themes: pd.DataFrame,
    relationships: pd.DataFrame,
    model: str,
) -> tuple[str, str]:
    """Generate a theme-classification narrative with GPT."""
    if themes.empty:
        return (
            "No theme candidates were generated for the current filtered inventory.",
            "No priority theme analysis is available until at least one theme passes the final criteria.",
        )

    theme_columns = [
        "Theme_ID",
        "Theme_Name",
        "Taxonomy_Alignment",
        "Business_Divisions",
        "GCRS_Values",
        "Risk_Count",
        "Material_Risks",
        "Non_Material_Risks",
        "Risk_IDs",
        "Common_Topic",
        "Key_Drivers",
        "Primary_Driver",
        "Primary_Metric",
        "Primary_Method",
        "Risk_Metrics",
        "Assessment_Methods",
        "Emerging_Indicator",
        "Cross_Business",
        "Review_Required",
        "Governance_Check",
        "Potential_Gap",
    ]
    theme_context = themes[[column for column in theme_columns if column in themes.columns]].head(40)
    relationship_context = relationships.head(80) if not relationships.empty else relationships
    system_prompt = (
        "You are a risk theme classification analyst. Use only the supplied records and computed facts. "
        "Existing Taxonomy_L1, Taxonomy_L2, and regulatory classifications are authoritative and must not be overwritten. "
        "Only discuss suggested risk themes, similar risk relationships, emerging clusters, and cross-business linkages. "
        "Explain grouping first through risk title and description semantics, then risk metric and Impact_Comment, "
        "then taxonomy and driver/root cause where the data supports it. "
        "Do not analyze, aggregate, or compare Impact_Numbers. Write concise Markdown."
    )
    user_prompt = f"""
Analyze these suggested risk themes for an executive risk inventory view.

Return exactly two Markdown sections using these headings:
## Theme Summary
- State how many suggested themes exist in the selected scope.
- Define and describe the main themes in business language.
- Summarize linkage and evidence using theme summaries, drivers, common topics, business divisions, GCRS, and sample Risk_IDs where useful.
- Do not mention similarity scores, confidence scores, embedding/vector scores, hybrid weights, or other model-scoring mechanics.

## Theme Analysis
- Suggest the 2-3 most important themes and support each with evidence such as driver, transmission channel, exposure, cross-business linkage, Risk_ID, GCRS, or Business_Division. If transmission, exposure, or managed-action evidence is not explicit, say so.
- Identify cross-division inconsistency hotspots for Risk_Metric and Assessment_Method, especially key Assessment_Method gaps.
- Add human-review priorities for mixed category, evidence quality, repeated themes, or governance concerns.
- Keep the language suitable for CRO and senior-management reporting; translate model evidence into business rationale.

Suggested theme mapping:
{theme_context.to_json(orient="records", force_ascii=False, indent=2)}

Similarity relationships:
{relationship_context.to_json(orient="records", force_ascii=False, indent=2)}

Filtered risk records:
{build_inventory_context(data)}
"""
    response = request_openai_text(model, system_prompt, user_prompt, 2600)
    return split_theme_ai_sections(response)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse strict JSON or a JSON object embedded in a model response."""
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def ask_inventory_chatbot(
    data: pd.DataFrame,
    question: str,
    model: str,
    chat_history: list[dict[str, str]] | None = None,
    prior_group_ids: list[str] | None = None,
    max_ids: int = CHAT_ASSOCIATED_RISK_LIMIT,
) -> tuple[str, list[str]]:
    """Answer an inventory question and return associated Group_ID values."""
    if data.empty:
        return "No risk records are available in the current sidebar filter scope.", []

    valid_ids = set(data["Group_ID"].astype(str).tolist())
    clean_prior_ids = [str(group_id) for group_id in (prior_group_ids or []) if str(group_id) in valid_ids]
    system_prompt = (
        "You are a risk inventory chatbot. Answer only from the provided inventory records. "
        "Do not invent records, IDs, counts, divisions, methods, metrics, or commentary. "
        "Do not analyze or aggregate Impact_Numbers. "
        "Use the recent conversation and previous associated Group_ID scope to resolve follow-up questions. "
        "Return JSON only with keys: answer_markdown, matching_group_ids. "
        "matching_group_ids must contain only Group_ID values from the provided records and should drive a filtered table. "
        f"Return up to {max_ids} matching_group_ids when the question describes a broad scope."
    )
    user_prompt = f"""
User question:
{question}

Recent chat history:
{format_chat_history(chat_history)}

Previous associated table Group_ID scope:
{json.dumps(clean_prior_ids[:max_ids], ensure_ascii=False)}

Filtered risk inventory records:
{build_inventory_context(data, limit=max_ids)}

Return JSON only:
{{
  "answer_markdown": "short Markdown answer grounded in the records",
  "matching_group_ids": ["Group_ID values for the risks most associated with the answer and current discussed scope"]
}}
"""
    raw = request_openai_text(model, system_prompt, user_prompt, 4000)
    payload = parse_json_object(raw)
    answer = str(payload.get("answer_markdown") or "").strip()
    if not answer:
        raise ValueError("OpenAI chatbot response did not include answer_markdown.")

    ids = [str(group_id) for group_id in payload.get("matching_group_ids", []) if str(group_id) in valid_ids]
    return answer, ids[:max_ids]
