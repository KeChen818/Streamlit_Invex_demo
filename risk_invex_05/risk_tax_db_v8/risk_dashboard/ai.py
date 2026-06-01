"""OpenAI integration and chatbot fallback logic."""

import json
import os
import re
from typing import Any

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .analysis import build_compare_fallback_analysis, build_taxonomy_fallback_summary
from .settings import AI_CONTEXT_COLUMNS, CHAT_ASSOCIATED_RISK_LIMIT, DEFAULT_MODEL
from .themes import build_theme_fallback_sections
from .utils import format_top_counts, top_counts


def openai_is_configured() -> bool:
    """Check whether live GPT analysis can run."""
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def selected_model() -> str:
    """Resolve the backend model without exposing it in the UI."""
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def build_inventory_context(data: pd.DataFrame, limit: int = 80) -> str:
    """Serialize filtered inventory records for GPT prompts."""
    columns = [column for column in AI_CONTEXT_COLUMNS if column in data.columns]
    records = data[columns].head(limit).to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False, indent=2)


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
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=max_output_tokens,
    )
    return response.output_text.strip()


def get_taxonomy_ai_summary(data: pd.DataFrame, group_summary: pd.DataFrame, model: str) -> str:
    """Generate the Taxonomy L1 narrative with GPT, falling back locally."""
    fallback = build_taxonomy_fallback_summary(data, group_summary)
    if not openai_is_configured():
        return fallback

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
{fallback}

Risk records:
{build_inventory_context(data)}
"""

    try:
        return request_openai_text(model, system_prompt, user_prompt, 1800)
    except Exception:
        return fallback + "\n\n_Local fallback used because live analysis is unavailable._"


def get_compare_ai_analysis(data: pd.DataFrame, model: str) -> str:
    """Generate the method/metric narrative with GPT, falling back locally."""
    fallback = build_compare_fallback_analysis(data)
    if not openai_is_configured():
        return fallback

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
{fallback}

Risk records:
{build_inventory_context(data)}
"""

    try:
        return request_openai_text(model, system_prompt, user_prompt, 2000)
    except Exception:
        return fallback + "\n\n_Local fallback used because live analysis is unavailable._"


def split_theme_ai_sections(text: str, fallback_summary: str, fallback_analysis: str) -> tuple[str, str]:
    """Split a two-section GPT theme response into dashboard card bodies."""
    clean = text.strip()
    summary_match = re.search(r"(?im)^#{1,3}\s*Theme Summary\s*$", clean)
    analysis_match = re.search(r"(?im)^#{1,3}\s*Theme Analysis\s*$", clean)
    if not summary_match or not analysis_match or analysis_match.start() <= summary_match.end():
        return fallback_summary, clean or fallback_analysis

    summary = clean[summary_match.end() : analysis_match.start()].strip()
    analysis = clean[analysis_match.end() :].strip()
    return summary or fallback_summary, analysis or fallback_analysis


def get_theme_ai_analysis(
    data: pd.DataFrame,
    themes: pd.DataFrame,
    relationships: pd.DataFrame,
    model: str,
) -> tuple[str, str]:
    """Generate a theme-classification narrative with GPT, falling back locally."""
    fallback_summary, fallback_analysis = build_theme_fallback_sections(themes, relationships)
    if not openai_is_configured():
        return fallback_summary, fallback_analysis
    if themes.empty:
        return fallback_summary, fallback_analysis

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

    try:
        response = request_openai_text(model, system_prompt, user_prompt, 2600)
        return split_theme_ai_sections(response, fallback_summary, fallback_analysis)
    except Exception:
        return fallback_summary, fallback_analysis + "\n\n_Local fallback used because live analysis is unavailable._"


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


def fallback_inventory_answer(
    data: pd.DataFrame,
    question: str,
    prior_group_ids: list[str] | None = None,
    max_ids: int = CHAT_ASSOCIATED_RISK_LIMIT,
) -> tuple[str, list[str]]:
    """Answer and filter by keyword matching when GPT chat is unavailable."""
    clean_question = question.strip().lower()
    valid_ids = set(data["Group_ID"].astype(str).tolist())
    clean_prior_ids = [str(group_id) for group_id in (prior_group_ids or []) if str(group_id) in valid_ids]
    contextual_terms = {
        "those",
        "these",
        "same",
        "previous",
        "above",
        "them",
        "they",
        "their",
        "scope",
        "within",
        "filter",
        "narrow",
        "also",
    }
    uses_prior_scope = bool(clean_prior_ids) and any(term in clean_question for term in contextual_terms)
    search_data = data[data["Group_ID"].astype(str).isin(clean_prior_ids)] if uses_prior_scope else data
    if search_data.empty:
        search_data = data

    search_columns = [column for column in AI_CONTEXT_COLUMNS if column in search_data.columns]
    searchable = search_data[search_columns].astype(str).agg(" ".join, axis=1).str.lower()

    tokens = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9/-]+", clean_question)
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "for",
            "with",
            "risk",
            "risks",
            "show",
            "what",
            "which",
            "use",
            "uses",
            "using",
            "are",
            "this",
            "that",
            "about",
        }
    ]
    if tokens:
        strict_mask = searchable.apply(lambda value: all(token in value for token in tokens))
        mask = strict_mask if strict_mask.any() else searchable.apply(
            lambda value: any(token in value for token in tokens)
        )
        matched = search_data[mask]
    else:
        matched = search_data

    if matched.empty:
        matched = search_data if uses_prior_scope else data

    group_counts = format_top_counts(top_counts(matched["Taxonomy_L1"], limit=3))
    metric_counts = format_top_counts(top_counts(matched["Risk_Metric"], limit=3))
    method_counts = format_top_counts(top_counts(matched["Assessment_Method"], limit=3))
    ids = matched["Group_ID"].astype(str).head(max_ids).tolist()
    scope_phrase = " from the prior discussed scope" if uses_prior_scope else ""
    answer = (
        f"I found **{len(matched)} associated risks**{scope_phrase} in the current filtered inventory. "
        f"Top Taxonomy L1 groups: {group_counts}. Top metrics: {metric_counts}. "
        f"Top methods: {method_counts}."
    )
    return answer, ids


def ask_inventory_chatbot(
    data: pd.DataFrame,
    question: str,
    model: str,
    chat_history: list[dict[str, str]] | None = None,
    prior_group_ids: list[str] | None = None,
    max_ids: int = CHAT_ASSOCIATED_RISK_LIMIT,
) -> tuple[str, list[str]]:
    """Answer an inventory question and return associated Group_ID values."""
    fallback_answer, fallback_ids = fallback_inventory_answer(data, question, prior_group_ids, max_ids)
    if not openai_is_configured():
        return fallback_answer, fallback_ids

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

    try:
        raw = request_openai_text(model, system_prompt, user_prompt, 4000)
        payload = parse_json_object(raw)
        answer = str(payload.get("answer_markdown") or "").strip() or fallback_answer
        ids = [str(group_id) for group_id in payload.get("matching_group_ids", []) if str(group_id) in valid_ids]
        if not ids:
            ids = fallback_ids
        return answer, ids[:max_ids]
    except Exception:
        return fallback_answer + "\n\n_Local fallback used because live chat is unavailable._", fallback_ids
