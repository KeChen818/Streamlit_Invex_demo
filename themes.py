"""AI-assisted risk theme classification utilities.

The official taxonomy columns remain source-of-truth fields. This module only
creates suggested risk-theme overlays, similarity relationships, and review
workflow fields for dashboard use.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import unique_join


THEME_REVIEW_ACTIONS = ["Pending Review", "Accept", "Rename", "Merge", "Split", "Reject"]

SIMILARITY_BANDS = [
    (0.90, "Near duplicate"),
    (0.82, "Same theme"),
    (0.72, "Related"),
    (0.00, "Weak relationship"),
]

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "within",
    "risk",
    "risks",
    "loss",
    "adverse",
    "impact",
    "affecting",
    "activities",
    "scenario",
    "assessment",
    "method",
    "metric",
    "taxonomy",
}

ACRONYMS = {
    "ai",
    "api",
    "ccp",
    "cre",
    "gcrs",
    "ib",
    "kri",
    "llm",
    "nfr",
    "qoq",
    "rwa",
    "sme",
    "wm",
}


def field_text(row: pd.Series, column: str) -> str:
    """Return a clean string field value from a risk row."""
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def build_semantic_payload(row: pd.Series) -> str:
    """Build a weighted semantic payload used for theme similarity.

    Titles are deliberately down-weighted because executive themes should be
    driven by risk driver, taxonomy, metric, method, and description patterns.
    """
    clean_title = re.sub(r"\bscenario\s*\d+\b", "", field_text(row, "Risk_Title"), flags=re.IGNORECASE)
    clean_title = re.sub(r"\b\d+\b", "", clean_title).strip()
    taxonomy = f'{field_text(row, "Taxonomy_L1")} / {field_text(row, "Taxonomy_L2")}'
    high_weight_sections = [
        field_text(row, "Taxonomy_L2"),
        field_text(row, "Risk_Metric"),
        field_text(row, "Assessment_Method"),
        field_text(row, "Risk_Exposure"),
    ]
    return f"""
Theme Signals:
{" ".join(high_weight_sections)}
{" ".join(high_weight_sections)}

Risk Name:
{clean_title}

Risk Description:
{field_text(row, "Risk_Description")}
{field_text(row, "Risk_Description")}

Risk Driver:
{field_text(row, "Risk_Exposure")}
{field_text(row, "Risk_Exposure")}

Business Division:
{field_text(row, "Business_Division")}

Taxonomy:
{taxonomy}
{taxonomy}
{taxonomy}

Risk Metric:
{field_text(row, "Risk_Metric")}
{field_text(row, "Risk_Metric")}
{field_text(row, "Risk_Metric")}

Assessment Method:
{field_text(row, "Assessment_Method")}
{field_text(row, "Assessment_Method")}
"""


def tokenize(text: str) -> list[str]:
    """Tokenize text for the local embedding fallback."""
    tokens = re.findall(r"[a-z0-9][a-z0-9/-]{1,}", str(text).lower())
    return [token for token in tokens if token not in STOP_WORDS and not token.isdigit()]


def token_bucket(token: str, dimensions: int) -> int:
    """Map a token to a stable hashed vector bucket."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dimensions


def build_local_embeddings(payloads: list[str], dimensions: int = 512) -> np.ndarray:
    """Create lightweight hashed TF-IDF embeddings for local MVP similarity."""
    tokenized = [tokenize(payload) for payload in payloads]
    document_count = max(len(tokenized), 1)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    vectors = np.zeros((len(payloads), dimensions), dtype=float)
    for row_index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for token, count in counts.items():
            bucket = token_bucket(token, dimensions)
            idf = math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0
            vectors[row_index, bucket] += (1.0 + math.log(count)) * idf

        norm = np.linalg.norm(vectors[row_index])
        if norm:
            vectors[row_index] = vectors[row_index] / norm

    return vectors


def calibrated_similarity(cosine_similarity: float) -> float:
    """Map local cosine similarity into the business threshold scale."""
    positive_similarity = max(float(cosine_similarity), 0.0)
    return round(min(1.0, 0.55 + (0.45 * positive_similarity)), 3)


def relation_for_score(score: float) -> str:
    """Classify a similarity score using the MVP threshold bands."""
    for threshold, label in SIMILARITY_BANDS:
        if score >= threshold:
            return label
    return "Weak relationship"


def build_similarity_relationships(
    data: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    relationship_threshold: float,
) -> pd.DataFrame:
    """Find top similar risks and return de-duplicated relationship records."""
    if data.empty or len(data) == 1:
        return pd.DataFrame()

    records = data.reset_index(drop=True)
    cosine_matrix = embeddings @ embeddings.T
    relationships: dict[tuple[str, str], dict[str, object]] = {}

    for source_index, source in records.iterrows():
        source_id = field_text(source, "Group_ID")
        candidate_indices = np.argsort(cosine_matrix[source_index])[::-1]
        matches = [idx for idx in candidate_indices if idx != source_index][:top_k]

        for target_index in matches:
            target = records.iloc[int(target_index)]
            target_id = field_text(target, "Group_ID")
            score = calibrated_similarity(cosine_matrix[source_index, target_index])
            if score < relationship_threshold:
                continue

            pair_key = tuple(sorted([source_id, target_id]))
            existing = relationships.get(pair_key)
            if existing and float(existing["Similarity_Score"]) >= score:
                continue

            relationships[pair_key] = {
                "Risk_ID": source_id,
                "Risk_Title": field_text(source, "Risk_Title"),
                "Similar_Risk_ID": target_id,
                "Similar_Risk_Title": field_text(target, "Risk_Title"),
                "Similarity_Score": score,
                "Relationship": relation_for_score(score),
                "Same_Taxonomy_L1": field_text(source, "Taxonomy_L1") == field_text(target, "Taxonomy_L1"),
                "Same_Metric": field_text(source, "Risk_Metric") == field_text(target, "Risk_Metric"),
                "Same_Method": field_text(source, "Assessment_Method")
                == field_text(target, "Assessment_Method"),
            }

    if not relationships:
        return pd.DataFrame()

    return pd.DataFrame(relationships.values()).sort_values(
        ["Similarity_Score", "Risk_ID"],
        ascending=[False, True],
    )


def find_parent(parent: dict[str, str], item: str) -> str:
    """Find a union-find parent with path compression."""
    if parent[item] != item:
        parent[item] = find_parent(parent, parent[item])
    return parent[item]


def union(parent: dict[str, str], left: str, right: str) -> None:
    """Union two risk IDs in the clustering helper."""
    left_parent = find_parent(parent, left)
    right_parent = find_parent(parent, right)
    if left_parent != right_parent:
        parent[right_parent] = left_parent


def cluster_risks(
    risk_ids: Iterable[str],
    relationships: pd.DataFrame,
    cluster_threshold: float,
) -> list[list[str]]:
    """Cluster risks by connecting pairs above the selected theme threshold."""
    parent = {risk_id: risk_id for risk_id in risk_ids}

    if not relationships.empty:
        strong_pairs = relationships[relationships["Similarity_Score"] >= cluster_threshold]
        for row in strong_pairs.to_dict("records"):
            left = str(row["Risk_ID"])
            right = str(row["Similar_Risk_ID"])
            if left in parent and right in parent:
                union(parent, left, right)

    clusters: dict[str, list[str]] = {}
    for risk_id in parent:
        clusters.setdefault(find_parent(parent, risk_id), []).append(risk_id)

    return sorted(
        [sorted(cluster) for cluster in clusters.values()],
        key=lambda values: (-len(values), values[0]),
    )


def initialize_centroids(embeddings: np.ndarray, cluster_count: int) -> np.ndarray:
    """Choose deterministic farthest-first seeds for target-count clustering."""
    selected = [0]
    similarity_to_selected = embeddings @ embeddings[0]

    while len(selected) < cluster_count:
        candidate_scores = similarity_to_selected.copy()
        candidate_scores[selected] = np.inf
        next_index = int(np.argmin(candidate_scores))
        if next_index in selected:
            break
        selected.append(next_index)
        similarity_to_selected = np.maximum(similarity_to_selected, embeddings @ embeddings[next_index])

    return embeddings[selected].copy()


def cluster_risks_to_target(
    risk_ids: Iterable[str],
    embeddings: np.ndarray,
    target_theme_count: int,
    max_iterations: int = 40,
) -> list[list[str]]:
    """Cluster risks into a target number of executive-level themes."""
    ids = list(risk_ids)
    if not ids:
        return []

    cluster_count = max(1, min(int(target_theme_count), len(ids)))
    if cluster_count == len(ids):
        return [[risk_id] for risk_id in ids]

    centroids = initialize_centroids(embeddings, cluster_count)
    cluster_count = len(centroids)
    labels = np.full(len(ids), -1, dtype=int)

    for _ in range(max_iterations):
        similarity = embeddings @ centroids.T
        next_labels = np.argmax(similarity, axis=1)

        empty_clusters = sorted(set(range(cluster_count)) - set(next_labels.tolist()))
        if empty_clusters:
            assigned_similarity = similarity[np.arange(len(ids)), next_labels]
            refill_candidates = np.argsort(assigned_similarity)
            used_candidates: set[int] = set()
            for empty_cluster in empty_clusters:
                for candidate in refill_candidates:
                    candidate_index = int(candidate)
                    if candidate_index not in used_candidates:
                        next_labels[candidate_index] = empty_cluster
                        used_candidates.add(candidate_index)
                        break

        if np.array_equal(labels, next_labels):
            break
        labels = next_labels

        updated_centroids = np.zeros_like(centroids)
        for cluster_index in range(cluster_count):
            members = embeddings[labels == cluster_index]
            if len(members) == 0:
                updated_centroids[cluster_index] = centroids[cluster_index]
                continue
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            updated_centroids[cluster_index] = centroid / norm if norm else centroid
        centroids = updated_centroids

    clusters = []
    for cluster_index in range(cluster_count):
        cluster = [ids[index] for index, label in enumerate(labels) if label == cluster_index]
        if cluster:
            clusters.append(sorted(cluster))

    return sorted(clusters, key=lambda values: (-len(values), values[0]))


def format_theme_label(value: str) -> str:
    """Convert taxonomy or metric text into a concise theme label."""
    clean = re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip()
    if not clean:
        return "Risk"

    words = []
    for word in clean.split():
        stripped = re.sub(r"[^A-Za-z0-9/-]", "", word)
        if stripped.lower() in ACRONYMS:
            words.append(stripped.upper())
        else:
            words.append(stripped.capitalize())
    label = " ".join(words)
    return label if label.lower().endswith("theme") else f"{label} Theme"


def dominant_value(series: pd.Series) -> str:
    """Return the most common non-empty value in a series."""
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    if clean.empty:
        return ""
    return str(clean.value_counts().idxmax())


def theme_gap_text(theme_data: pd.DataFrame) -> str:
    """Summarize potential metric or method gaps across divisions in a theme."""
    if theme_data["Business_Division"].nunique() < 2:
        return "Single-division theme; no cross-division gap flagged."

    metric_counts = theme_data.groupby("Business_Division")["Risk_Metric"].nunique()
    method_counts = theme_data.groupby("Business_Division")["Assessment_Method"].nunique()
    signals = []
    if theme_data["Risk_Metric"].nunique() > 1:
        signals.append(
            "metric variation across "
            + ", ".join(metric_counts[metric_counts > 0].index.astype(str).tolist())
        )
    if theme_data["Assessment_Method"].nunique() > 1:
        signals.append(
            "method variation across "
            + ", ".join(method_counts[method_counts > 0].index.astype(str).tolist())
        )
    return "; ".join(signals) if signals else "No metric or method gap flagged."


def average_internal_similarity(cluster: list[str], relationships: pd.DataFrame) -> float:
    """Calculate average pair similarity inside a suggested theme cluster."""
    if len(cluster) < 2 or relationships.empty:
        return 0.0

    cluster_set = set(cluster)
    internal = relationships[
        relationships["Risk_ID"].isin(cluster_set) & relationships["Similar_Risk_ID"].isin(cluster_set)
    ]
    if internal.empty:
        return 0.0
    return round(float(internal["Similarity_Score"].mean()), 3)


def build_theme_records(
    data: pd.DataFrame,
    clusters: list[list[str]],
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """Create AI-enriched theme mapping records from clusters."""
    theme_records = []
    for index, cluster in enumerate(clusters, start=1):
        theme_data = data[data["Group_ID"].astype(str).isin(cluster)].copy()
        dominant_l2 = dominant_value(theme_data["Taxonomy_L2"])
        dominant_metric = dominant_value(theme_data["Risk_Metric"])
        dominant_l1 = dominant_value(theme_data["Taxonomy_L1"])
        label_source = dominant_l2 or dominant_metric or dominant_l1
        theme_name = format_theme_label(label_source)
        taxonomies = sorted(
            {
                value
                for value in theme_data[["Taxonomy_L0", "Taxonomy_L1"]].astype(str).stack().tolist()
                if value and value.lower() != "nan"
            }
        )
        divisions = sorted(theme_data["Business_Division"].dropna().astype(str).unique().tolist())
        metrics = unique_join(theme_data["Risk_Metric"], limit=4)
        methods = unique_join(theme_data["Assessment_Method"], limit=3)
        drivers = unique_join(theme_data["Taxonomy_L2"], limit=3)
        avg_similarity = average_internal_similarity(cluster, relationships)
        cross_business = len(divisions) > 1
        emerging = bool(cross_business and len(cluster) >= 2 and theme_data["Taxonomy_L1"].nunique() > 1)

        theme_records.append(
            {
                "Theme_ID": f"THM_{index:03d}",
                "Theme_Name": theme_name,
                "Theme_Summary": (
                    f"Risks related to {drivers}. Business divisions include "
                    f"{', '.join(divisions) if divisions else 'none'}. Metrics include {metrics}; "
                    f"assessment methods include {methods}."
                ),
                "Taxonomy_Alignment": taxonomies,
                "Business_Divisions": divisions,
                "Risk_Count": int(len(theme_data)),
                "Risk_IDs": cluster,
                "Embedding_Centroid_ID": f"vec_{index:03d}",
                "LLM_Generated": False,
                "Human_Reviewed": False,
                "Review_Action": "Pending Review",
                "Reviewed_By": "",
                "Review_Date": "",
                "Human_Notes": "",
                "Created_Quarter": dominant_value(theme_data["Reporting_Quarter"]),
                "Emerging_Indicator": emerging,
                "Cross_Business": cross_business,
                "Average_Similarity": avg_similarity,
                "Potential_Gap": theme_gap_text(theme_data),
            }
        )

    return pd.DataFrame(theme_records)


def build_quarter_theme_summary(data: pd.DataFrame, themes: pd.DataFrame) -> pd.DataFrame:
    """Create a quarter-by-theme count table for QoQ comparison."""
    if themes.empty:
        return pd.DataFrame()

    rows = []
    for theme in themes.to_dict("records"):
        risk_ids = set(theme["Risk_IDs"])
        theme_data = data[data["Group_ID"].astype(str).isin(risk_ids)]
        counts = theme_data["Reporting_Quarter"].fillna("Unknown").astype(str).value_counts()
        for quarter, count in counts.items():
            rows.append(
                {
                    "Theme_ID": theme["Theme_ID"],
                    "Theme_Name": theme["Theme_Name"],
                    "Reporting_Quarter": quarter,
                    "Risk_Count": int(count),
                }
            )

    return pd.DataFrame(rows).sort_values(["Reporting_Quarter", "Theme_ID"])


def build_theme_classification(
    data: pd.DataFrame,
    top_k: int = 5,
    relationship_threshold: float = 0.72,
    cluster_threshold: float = 0.82,
    target_theme_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the local MVP theme classification pipeline."""
    scoped = data.reset_index(drop=True).copy()
    payloads = [build_semantic_payload(row) for _, row in scoped.iterrows()]
    embeddings = build_local_embeddings(payloads)
    relationships = build_similarity_relationships(scoped, embeddings, top_k, relationship_threshold)
    risk_ids = scoped["Group_ID"].astype(str).tolist()
    if target_theme_count is not None and target_theme_count < len(risk_ids):
        clusters = cluster_risks_to_target(risk_ids, embeddings, target_theme_count)
    else:
        clusters = cluster_risks(risk_ids, relationships, cluster_threshold)
    themes = build_theme_records(scoped, clusters, relationships)
    quarter_summary = build_quarter_theme_summary(scoped, themes)
    return themes, relationships, quarter_summary


def build_theme_fallback_analysis(themes: pd.DataFrame, relationships: pd.DataFrame) -> str:
    """Create a deterministic narrative when live GPT analysis is unavailable."""
    if themes.empty:
        return "No theme candidates were generated for the current filtered inventory."

    cross_business = int(themes["Cross_Business"].sum())
    emerging = int(themes["Emerging_Indicator"].sum())
    top_themes = themes.sort_values(["Risk_Count", "Average_Similarity"], ascending=[False, False]).head(5)
    top_lines = [
        f"- **{row.Theme_Name}**: {int(row.Risk_Count)} risks; {row.Potential_Gap}"
        for row in top_themes.itertuples(index=False)
    ]
    near_duplicates = 0
    if not relationships.empty:
        near_duplicates = int((relationships["Relationship"] == "Near duplicate").sum())

    return "\n".join(
        [
            f"Generated **{len(themes)} suggested risk themes** for the current inventory.",
            f"Cross-business themes: **{cross_business}**. Emerging clusters: **{emerging}**. Near-duplicate relationships: **{near_duplicates}**.",
            "",
            "Priority review themes:",
            *top_lines,
        ]
    )
