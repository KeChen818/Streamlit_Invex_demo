import json
import re
from typing import List

import pandas as pd


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with",
    "risk", "risks", "show", "give", "summarize", "compare", "please",
    "what", "how", "which", "is", "are", "by", "about"
}


def tokenize(text: str) -> List[str]:
    text = str(text or "").lower()
    tokens = re.findall(r"[a-zA-Z0-9_]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def infer_intent(prompt: str) -> str:
    p = (prompt or "").lower()

    if any(x in p for x in ["qoq", "quarter over quarter", "change", "moved", "movement"]):
        return "qoq_compare"

    if any(x in p for x in ["material", "materiality", "non-material", "non material"]):
        return "materiality_review"

    if any(x in p for x in ["taxonomy", "l0", "l1", "l2", "category", "risk type"]):
        return "taxonomy_mapping"

    if any(x in p for x in ["executive", "committee", "senior management", "brief"]):
        return "executive_brief"

    return "general"


def pick_relevant_rows(df: pd.DataFrame, prompt: str, k: int = 35) -> pd.DataFrame:
    if df.empty:
        return df

    q = set(tokenize(prompt))

    if not q:
        return df.head(k)

    searchable_cols = [
        "group_id",
        "risk_name",
        "risk_description",
        "business_division",
        "legal_entity",
        "risk_type",
        "taxonomy_10",
        "taxonomy_11",
        "taxonomy_12",
        "overall_materiality",
        "risk_status",
        "assessment_method",
    ]

    searchable_cols = [c for c in searchable_cols if c in df.columns]

    tmp = df.copy()

    def score(row):
        blob = " ".join(str(row.get(c, "")) for c in searchable_cols)
        row_tokens = set(tokenize(blob))
        return len(q.intersection(row_tokens))

    tmp["_rel_score"] = tmp.apply(score, axis=1)

    if tmp["_rel_score"].max() == 0:
        return tmp.drop(columns=["_rel_score"]).head(k)

    return (
        tmp.sort_values("_rel_score", ascending=False)
        .drop(columns=["_rel_score"])
        .head(k)
    )


def select_context_columns(df: pd.DataFrame, intent: str) -> list[str]:
    base_cols = [
        "group_id",
        "risk_name",
        "risk_description",
        "business_division",
        "legal_entity",
        "risk_type",
        "taxonomy_10",
        "taxonomy_11",
        "taxonomy_12",
        "overall_materiality",
        "risk_status",
        "assessment_method",
        "likelihood",
        "impact",
    ]

    qoq_cols = [
        "previous_overall_materiality",
        "current_overall_materiality",
        "previous_likelihood",
        "current_likelihood",
        "previous_impact",
        "current_impact",
        "qoq_change",
        "change_rationale",
    ]

    if intent == "qoq_compare":
        cols = base_cols + qoq_cols
    else:
        cols = base_cols

    return [c for c in cols if c in df.columns]


def build_context(df: pd.DataFrame, prompt: str, max_rows: int = 35) -> str:
    intent = infer_intent(prompt)

    relevant = pick_relevant_rows(df, prompt, k=max_rows)
    cols = select_context_columns(relevant, intent)

    scope_summary = {
        "rows_in_current_filter": int(len(df)),
        "rows_sent_to_model": int(len(relevant)),
        "material": int((df["overall_materiality"] == "Material").sum())
        if "overall_materiality" in df.columns else None,
        "non_material": int((df["overall_materiality"] == "Non-Material").sum())
        if "overall_materiality" in df.columns else None,
        "intent": intent,
    }

    payload = {
        "scope_summary": scope_summary,
        "records": relevant[cols].fillna("").to_dict("records") if cols else [],
    }

    return json.dumps(payload, indent=2)
