"""AI-assisted risk theme classification utilities.

The official taxonomy columns remain source-of-truth fields. This module only
creates suggested risk-theme overlays, similarity relationships, and review
workflow fields for dashboard use.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import unique_join

try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        Document = None

try:
    from langchain_community.vectorstores import FAISS
except Exception:
    FAISS = None

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None


THEME_REVIEW_ACTIONS = ["Pending Review", "Accept", "Rename", "Merge", "Split", "Reject"]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
MIN_THEME_CONFIDENCE = 0.50
HYBRID_WEIGHTS = {
    "semantic": 0.50,
    "metric_impact": 0.25,
    "taxonomy": 0.15,
    "driver": 0.10,
}
_OPENAI_FAISS_EMBEDDING_CACHE: dict[str, np.ndarray] = {}

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
    "variant",
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

ROOT_CAUSE_LEVEL_COLUMNS = [
    "Root_Cause_L0",
    "Root_Cause_L1",
]

ROOT_CAUSE_DETAIL_COLUMNS = [
    "Root_Cause_Comment",
    "Early_Warning_Sign(EWS)",
]

ROOT_CAUSE_FALLBACK_COLUMNS = [
    "Risk_Exposure",
]


def field_text(row: pd.Series, column: str) -> str:
    """Return a clean string field value from a risk row."""
    value = row.get(column, "")
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def root_cause_level(row: pd.Series) -> str:
    """Return the L0/L1 root-cause classification for theme grouping."""
    values = [field_text(row, column) for column in ROOT_CAUSE_LEVEL_COLUMNS]
    values = [value for value in values if value]
    return " / ".join(dict.fromkeys(values))


def root_cause_driver(row: pd.Series) -> str:
    """Return root-cause context for embeddings and theme detail summaries."""
    values = [root_cause_level(row)]
    values.extend(field_text(row, column) for column in ROOT_CAUSE_DETAIL_COLUMNS)
    values = [value for value in values if value]
    if values:
        return " | ".join(dict.fromkeys(values))

    for column in ROOT_CAUSE_FALLBACK_COLUMNS:
        value = field_text(row, column)
        if value:
            return value
    return ""


def root_cause_evidence(row: pd.Series) -> str:
    """Return comment/EWS evidence without using it as a theme split key."""
    values = [field_text(row, column) for column in ROOT_CAUSE_DETAIL_COLUMNS]
    values = [value for value in values if value]
    if values:
        return " | ".join(dict.fromkeys(values))

    for column in ROOT_CAUSE_FALLBACK_COLUMNS:
        value = field_text(row, column)
        if value:
            return value
    return ""


def clean_risk_title(row: pd.Series) -> str:
    """Remove scenario numbering from a risk title before semantic matching."""
    clean_title = re.sub(r"\bscenario\s*\d+\b", "", field_text(row, "Risk_Title"), flags=re.IGNORECASE)
    return re.sub(r"\b\d+\b", "", clean_title).strip()


def risk_identifier(row: pd.Series) -> str:
    """Return the source risk identifier used in vector metadata."""
    return field_text(row, "Group_ID")


def build_standard_risk_text(row: pd.Series) -> str:
    """Create a clean full-risk text block for traceability and summaries."""
    taxonomy = f'{field_text(row, "Taxonomy_L1")} / {field_text(row, "Taxonomy_L2")}'
    return f"""
Risk ID: {risk_identifier(row)}

Primary Risk Name:
{clean_risk_title(row)}

Primary Risk Description:
{field_text(row, "Risk_Description")}

Risk Metric:
{field_text(row, "Risk_Metric")}

Impact Commentary:
{field_text(row, "Impact_Comment")}

Taxonomy:
{taxonomy}

Risk Driver:
{root_cause_driver(row)}

Business Division:
{field_text(row, "Business_Division")}

GCRS:
{field_text(row, "GCRS")}

Assessment Method:
{field_text(row, "Assessment_Method")}
"""


def build_semantic_payload(row: pd.Series) -> str:
    """Build a weighted semantic payload used for theme similarity.

    Risk title and description are deliberately highest-weight because theme
    grouping should start with the risk title and description, not only official taxonomy.
    """
    clean_title = clean_risk_title(row)
    taxonomy = f'{field_text(row, "Taxonomy_L1")} / {field_text(row, "Taxonomy_L2")}'
    description = field_text(row, "Risk_Description")
    metric_impact = f'{field_text(row, "Risk_Metric")} {field_text(row, "Impact_Comment")}'
    root_cause = root_cause_driver(row)
    return f"""
Primary Risk Title:
{clean_title}
{clean_title}

Primary Risk Description:
{description}
{description}
{description}

Risk Metric and Impact Commentary:
{metric_impact}
{metric_impact}

Taxonomy Context:
{taxonomy}

Root Cause Driver:
{root_cause}
"""


def tokenize(text: str) -> list[str]:
    """Tokenize text for the local embedding fallback."""
    tokens = re.findall(r"[a-z0-9][a-z0-9/-]{1,}", str(text).lower())
    return [token for token in tokens if token not in STOP_WORDS and not token.isdigit()]


def semantic_keywords(text: str, limit: int = 6) -> str:
    """Return stable non-numeric keywords for root-cause/description signatures."""
    tokens = [
        token
        for token in tokenize(text)
        if len(token) > 2 and not re.search(r"\d", token)
    ]
    if not tokens:
        return ""
    return " ".join(token for token, _ in Counter(tokens).most_common(limit))


def title_description_tokens(row: pd.Series) -> list[str]:
    """Tokenize title and description for shared topic extraction."""
    text = f'{clean_risk_title(row)} {field_text(row, "Risk_Description")}'
    return [
        token
        for token in tokenize(text)
        if len(token) > 2 and not re.search(r"\d", token)
    ]


def title_description_sections(row: pd.Series) -> list[list[str]]:
    """Tokenize title and description separately to avoid cross-boundary phrases."""
    sections = [clean_risk_title(row), field_text(row, "Risk_Description")]
    return [
        [token for token in tokenize(section) if len(token) > 2 and not re.search(r"\d", token)]
        for section in sections
    ]


def common_topic(theme_data: pd.DataFrame, limit: int = 4) -> str:
    """Extract a repeated title/description topic from a group of risks."""
    if theme_data.empty or len(theme_data) < 2:
        return ""

    rows = [row for _, row in theme_data.iterrows()]
    documents = []
    for row in rows:
        tokens = title_description_tokens(row)
        if tokens:
            documents.append((row, tokens))
    docs = [tokens for _, tokens in documents]
    if len(docs) < 2:
        return ""

    minimum_documents = max(2, math.ceil(len(docs) * 0.30))
    token_document_frequency: Counter[str] = Counter()
    phrase_document_frequency: Counter[str] = Counter()
    for row, tokens in documents:
        token_document_frequency.update(set(tokens))
        phrases = set()
        for section in title_description_sections(row):
            for width in (3, 2):
                for index in range(max(len(section) - width + 1, 0)):
                    phrase_tokens = section[index : index + width]
                    if len(set(phrase_tokens)) != len(phrase_tokens):
                        continue
                    phrase = " ".join(phrase_tokens)
                    if not any(piece in STOP_WORDS for piece in phrase_tokens):
                        phrases.add(phrase)
        phrase_document_frequency.update(phrases)

    phrase_candidates = [
        (phrase, count)
        for phrase, count in phrase_document_frequency.items()
        if count >= minimum_documents
    ]
    if phrase_candidates:
        phrase_candidates.sort(key=lambda item: (item[1], len(item[0].split()), item[0]), reverse=True)
        return phrase_candidates[0][0]

    token_candidates = [
        (token, count)
        for token, count in token_document_frequency.items()
        if count >= minimum_documents
    ]
    token_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return " ".join(token for token, _ in token_candidates[:limit])


def topic_adds_signal(topic: str, base_text: str) -> bool:
    """Check whether a common topic adds information beyond taxonomy/root-cause."""
    topic_tokens = set(tokenize(topic))
    base_tokens = set(tokenize(base_text))
    return bool(topic_tokens - base_tokens)


def taxonomy_l2_is_broad(value: str) -> bool:
    """Identify generic taxonomy values where title/description should refine themes."""
    clean = str(value or "").strip().lower()
    if not clean:
        return True
    return clean in {
        "general",
        "miscellaneous",
        "not mapped",
        "other",
        "unmapped",
        "operational risk",
        "business risk",
        "emerging risk",
    }


def topic_should_split_theme(taxonomy_l2: str, root_keywords: str, topic: str, base_text: str) -> bool:
    """Use common topic as a split key only when it is stable and needed."""
    if not topic or not topic_adds_signal(topic, base_text):
        return False
    return bool(root_keywords) or taxonomy_l2_is_broad(taxonomy_l2)


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


def normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize embedding rows so dot product is cosine similarity."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_langchain_documents(data: pd.DataFrame, payloads: list[str]) -> list[object]:
    """Build weighted LangChain documents with risk metadata for FAISS indexing."""
    if Document is None:
        return []

    docs = []
    for position, (_, row) in enumerate(data.iterrows()):
        docs.append(
            Document(
                page_content=payloads[position],
                metadata={
                    "risk_id": risk_identifier(row),
                    "risk_title": clean_risk_title(row),
                    "taxonomy_l1": field_text(row, "Taxonomy_L1"),
                    "taxonomy_l2": field_text(row, "Taxonomy_L2"),
                    "business": field_text(row, "Business_Division"),
                    "gcrs": field_text(row, "GCRS"),
                },
            )
        )
    return docs


def build_openai_faiss_embeddings(data: pd.DataFrame, payloads: list[str]) -> np.ndarray | None:
    """Embed weighted risk payloads with OpenAI and load them into FAISS when available."""
    if not os.getenv("OPENAI_API_KEY") or OpenAIEmbeddings is None or FAISS is None or Document is None:
        return None

    documents = build_langchain_documents(data, payloads)
    if not documents:
        return None

    try:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL
        cache_payload = model_name + "\n" + "\n---risk-doc---\n".join(doc.page_content for doc in documents)
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
        if cache_key in _OPENAI_FAISS_EMBEDDING_CACHE:
            return _OPENAI_FAISS_EMBEDDING_CACHE[cache_key].copy()

        embedding_model = OpenAIEmbeddings(
            model=model_name
        )
        vector_store = FAISS.from_documents(documents, embedding_model)
        vectors = [
            vector_store.index.reconstruct(index)
            for index in range(vector_store.index.ntotal)
        ]
        if len(vectors) != len(documents):
            return None
        embedding_matrix = normalize_embedding_matrix(np.asarray(vectors, dtype=float))
        _OPENAI_FAISS_EMBEDDING_CACHE[cache_key] = embedding_matrix.copy()
        return embedding_matrix
    except Exception:
        return None


def build_embedding_layer(data: pd.DataFrame, payloads: list[str]) -> np.ndarray:
    """Use OpenAI/FAISS embeddings when configured, otherwise use local embeddings."""
    openai_embeddings = build_openai_faiss_embeddings(data, payloads)
    if openai_embeddings is not None:
        return openai_embeddings
    return build_local_embeddings(payloads)


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


def token_similarity(left: str, right: str) -> float:
    """Return a simple Jaccard score for two semantic text fields."""
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def taxonomy_alignment_score(source: pd.Series, target: pd.Series) -> float:
    """Score whether two risks align to the same official taxonomy levels."""
    source_l2 = field_text(source, "Taxonomy_L2")
    target_l2 = field_text(target, "Taxonomy_L2")
    if source_l2 and source_l2 == target_l2:
        return 1.0

    source_l1 = field_text(source, "Taxonomy_L1")
    target_l1 = field_text(target, "Taxonomy_L1")
    if source_l1 and source_l1 == target_l1:
        return 0.75

    source_l0 = field_text(source, "Taxonomy_L0")
    target_l0 = field_text(target, "Taxonomy_L0")
    if source_l0 and source_l0 == target_l0:
        return 0.45
    return 0.0


def driver_similarity_score(source: pd.Series, target: pd.Series) -> float:
    """Score common risk driver/root-cause patterns beyond similar wording."""
    source_driver_level = root_cause_level(source)
    target_driver_level = root_cause_level(target)
    if source_driver_level and source_driver_level == target_driver_level:
        return 1.0
    return token_similarity(root_cause_driver(source), root_cause_driver(target))


def metric_impact_similarity_score(source: pd.Series, target: pd.Series) -> float:
    """Score common risk metric and impact-comment assumption text."""
    source_metric = field_text(source, "Risk_Metric")
    target_metric = field_text(target, "Risk_Metric")
    if source_metric and target_metric and source_metric == target_metric:
        metric_score = 1.0
    else:
        metric_score = token_similarity(source_metric, target_metric)

    impact_score = token_similarity(
        field_text(source, "Impact_Comment"),
        field_text(target, "Impact_Comment"),
    )
    if source_metric or target_metric:
        return round((0.60 * metric_score) + (0.40 * impact_score), 3)
    return round(impact_score, 3)


def hybrid_similarity_components(
    source: pd.Series,
    target: pd.Series,
    semantic_similarity: float,
) -> dict[str, float]:
    """Blend title/description, metric/impact, taxonomy, and driver signals."""
    metric_impact_score = metric_impact_similarity_score(source, target)
    taxonomy_score = taxonomy_alignment_score(source, target)
    driver_score = driver_similarity_score(source, target)
    hybrid_score = (
        HYBRID_WEIGHTS["semantic"] * semantic_similarity
        + HYBRID_WEIGHTS["metric_impact"] * metric_impact_score
        + HYBRID_WEIGHTS["taxonomy"] * taxonomy_score
        + HYBRID_WEIGHTS["driver"] * driver_score
    )
    return {
        "Similarity_Score": round(float(hybrid_score), 3),
        "Semantic_Similarity": round(float(semantic_similarity), 3),
        "Metric_Impact_Similarity": round(float(metric_impact_score), 3),
        "Taxonomy_Alignment_Score": round(float(taxonomy_score), 3),
        "Driver_Similarity": round(float(driver_score), 3),
    }


def build_hybrid_similarity_matrix(
    data: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, float]]]:
    """Create the hybrid score matrix used for relationship search and clustering."""
    records = data.reset_index(drop=True)
    cosine_matrix = embeddings @ embeddings.T
    score_matrix = np.eye(len(records), dtype=float)
    component_lookup: dict[tuple[int, int], dict[str, float]] = {}

    for source_index, source in records.iterrows():
        for target_index in range(source_index + 1, len(records)):
            target = records.iloc[target_index]
            semantic_score = calibrated_similarity(cosine_matrix[source_index, target_index])
            components = hybrid_similarity_components(source, target, semantic_score)
            score = components["Similarity_Score"]
            score_matrix[source_index, target_index] = score
            score_matrix[target_index, source_index] = score
            component_lookup[(source_index, target_index)] = components
            component_lookup[(target_index, source_index)] = components

    return score_matrix, component_lookup


def build_similarity_relationships(
    data: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int,
    relationship_threshold: float,
    component_lookup: dict[tuple[int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Find top hybrid-similar risks and return de-duplicated relationship records."""
    if data.empty or len(data) == 1:
        return pd.DataFrame()

    records = data.reset_index(drop=True)
    relationships: dict[tuple[str, str], dict[str, object]] = {}
    component_lookup = component_lookup or {}

    for source_index, source in records.iterrows():
        source_id = field_text(source, "Group_ID")
        candidate_indices = np.argsort(similarity_matrix[source_index])[::-1]
        matches = [idx for idx in candidate_indices if idx != source_index][:top_k]

        for target_index in matches:
            target = records.iloc[int(target_index)]
            target_id = field_text(target, "Group_ID")
            score = round(float(similarity_matrix[source_index, target_index]), 3)
            if score < relationship_threshold:
                continue

            pair_key = tuple(sorted([source_id, target_id]))
            existing = relationships.get(pair_key)
            if existing and float(existing["Similarity_Score"]) >= score:
                continue

            components = component_lookup.get((source_index, int(target_index)), {})
            relationship_record = {
                "Risk_ID": source_id,
                "Risk_Title": field_text(source, "Risk_Title"),
                "Similar_Risk_ID": target_id,
                "Similar_Risk_Title": field_text(target, "Risk_Title"),
                "Similarity_Score": score,
                "Confidence_Score": score,
                "Relationship": relation_for_score(score),
                "Same_Taxonomy_L1": field_text(source, "Taxonomy_L1") == field_text(target, "Taxonomy_L1"),
                "Mixed_Category_Warning": field_text(source, "Taxonomy_L1") != field_text(target, "Taxonomy_L1"),
                "Same_Metric": field_text(source, "Risk_Metric") == field_text(target, "Risk_Metric"),
                "Same_Method": field_text(source, "Assessment_Method")
                == field_text(target, "Assessment_Method"),
            }
            relationship_record.update(components)
            relationships[pair_key] = relationship_record

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


def initialize_medoids(similarity_matrix: np.ndarray, cluster_count: int) -> list[int]:
    """Choose deterministic farthest-first medoids from a similarity matrix."""
    selected = [0]
    similarity_to_selected = similarity_matrix[:, 0].copy()

    while len(selected) < cluster_count:
        candidate_scores = similarity_to_selected.copy()
        candidate_scores[selected] = np.inf
        next_index = int(np.argmin(candidate_scores))
        if next_index in selected:
            break
        selected.append(next_index)
        similarity_to_selected = np.maximum(similarity_to_selected, similarity_matrix[:, next_index])

    return selected


def cluster_risks_to_target_by_similarity(
    risk_ids: Iterable[str],
    similarity_matrix: np.ndarray,
    target_theme_count: int,
    max_iterations: int = 40,
) -> list[list[str]]:
    """Cluster risks into a target count using hybrid similarity, not wording alone."""
    ids = list(risk_ids)
    if not ids:
        return []

    cluster_count = max(1, min(int(target_theme_count), len(ids)))
    if cluster_count == len(ids):
        return [[risk_id] for risk_id in ids]

    medoids = initialize_medoids(similarity_matrix, cluster_count)
    labels = np.full(len(ids), -1, dtype=int)

    for _ in range(max_iterations):
        medoid_similarity = similarity_matrix[:, medoids]
        next_labels = np.argmax(medoid_similarity, axis=1)

        empty_clusters = sorted(set(range(len(medoids))) - set(next_labels.tolist()))
        if empty_clusters:
            assigned_similarity = medoid_similarity[np.arange(len(ids)), next_labels]
            refill_candidates = np.argsort(assigned_similarity)
            used_candidates: set[int] = set()
            for empty_cluster in empty_clusters:
                for candidate in refill_candidates:
                    candidate_index = int(candidate)
                    if candidate_index not in used_candidates and candidate_index not in medoids:
                        next_labels[candidate_index] = empty_cluster
                        medoids[empty_cluster] = candidate_index
                        used_candidates.add(candidate_index)
                        break

        next_medoids = medoids.copy()
        for cluster_index in range(len(medoids)):
            members = np.where(next_labels == cluster_index)[0]
            if len(members) == 0:
                continue
            cluster_similarity = similarity_matrix[np.ix_(members, members)]
            next_medoids[cluster_index] = int(members[np.argmax(cluster_similarity.mean(axis=1))])

        if np.array_equal(labels, next_labels) and next_medoids == medoids:
            break
        labels = next_labels
        medoids = next_medoids

    clusters = []
    for cluster_index in range(len(medoids)):
        cluster = [ids[index] for index, label in enumerate(labels) if label == cluster_index]
        if cluster:
            clusters.append(sorted(cluster))

    return sorted(clusters, key=lambda values: (-len(values), values[0]))


def normalize_signature(value: str) -> str:
    """Normalize a field into a stable duplicate-theme signature."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def cluster_theme_signature(data: pd.DataFrame, cluster: list[str]) -> str:
    """Build the taxonomy/root-cause signature used to merge duplicate themes."""
    theme_data = data[data["Group_ID"].astype(str).isin(cluster)]
    taxonomy_l1 = dominant_value(theme_data["Taxonomy_L1"])
    taxonomy_l2 = dominant_value(theme_data["Taxonomy_L2"])
    root_cause_text = " ".join(theme_data.apply(root_cause_level, axis=1).tolist())
    root_keywords = semantic_keywords(root_cause_text, limit=5)
    shared_topic = common_topic(theme_data)
    if taxonomy_l1 or taxonomy_l2:
        signature_parts = [taxonomy_l1, taxonomy_l2]
        if root_keywords:
            signature_parts.append(root_keywords)
        if topic_should_split_theme(taxonomy_l2, root_keywords, shared_topic, "|".join(signature_parts)):
            signature_parts.append(shared_topic)
        return normalize_signature("|".join(signature_parts))

    fallback = "|".join(
        [
            root_keywords,
            shared_topic,
            dominant_value(theme_data["Business_Division"]),
        ]
    )
    return normalize_signature(fallback) or "|".join(cluster)


def merge_duplicate_theme_clusters(data: pd.DataFrame, clusters: list[list[str]]) -> list[list[str]]:
    """Merge clusters that map to the same suggested theme signature."""
    merged: dict[str, list[str]] = {}
    for cluster in clusters:
        signature = cluster_theme_signature(data, cluster)
        merged.setdefault(signature, []).extend(cluster)

    unique_clusters = [sorted(set(cluster)) for cluster in merged.values()]
    return sorted(unique_clusters, key=lambda values: (-len(values), values[0]))


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


def unique_values(series: pd.Series, limit: int = 6) -> list[str]:
    """Return ordered unique non-empty values for theme evidence fields."""
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    return clean.drop_duplicates().head(limit).tolist()


def title_fragment(value: str, max_words: int = 4) -> str:
    """Create a compact title-cased fragment from root-cause keywords."""
    keywords = semantic_keywords(value, limit=max_words)
    if not keywords:
        return ""
    return " ".join(word.upper() if word in ACRONYMS else word.capitalize() for word in keywords.split())


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


def materiality_counts(theme_data: pd.DataFrame) -> tuple[int, int]:
    """Count material/non-material risks using the source Overall_Materiality flag."""
    material_count = int(theme_data["Is_Material"].sum()) if "Is_Material" in theme_data else 0
    non_material_count = int(len(theme_data) - material_count)
    return material_count, non_material_count


def theme_key_drivers(theme_data: pd.DataFrame, shared_topic: str) -> list[str]:
    """Identify concise driver labels for theme inventory and review."""
    drivers = []
    for value in theme_data.apply(root_cause_level, axis=1).dropna().astype(str).tolist():
        clean = re.sub(r"\s+", " ", value).strip()
        if clean and clean not in drivers:
            drivers.append(clean)
    if shared_topic and shared_topic not in drivers:
        drivers.append(shared_topic)
    return drivers[:5]


def theme_review_flag(theme_data: pd.DataFrame, confidence_score: float) -> tuple[bool, str]:
    """Flag mixed or low-confidence themes for human governance review."""
    mixed_category = theme_data["Taxonomy_L1"].nunique() > 1 or theme_data["Taxonomy_L2"].nunique() > 4
    if mixed_category and confidence_score < 0.78:
        return True, "Review required: mixed taxonomy alignment and supporting evidence should be validated."
    if mixed_category:
        return True, "Review required: mixed taxonomy alignment."
    if confidence_score < 0.72:
        return True, "Review required: supporting evidence should be validated."
    return False, "Standard review: taxonomy authority preserved."


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


def average_internal_similarity_from_matrix(
    cluster: list[str],
    id_to_index: dict[str, int],
    similarity_matrix: np.ndarray,
) -> float:
    """Calculate average internal similarity from the full hybrid score matrix."""
    indices = [id_to_index[risk_id] for risk_id in cluster if risk_id in id_to_index]
    if len(indices) < 2:
        return 0.0
    values = []
    for left_position, left_index in enumerate(indices):
        for right_index in indices[left_position + 1 :]:
            values.append(float(similarity_matrix[left_index, right_index]))
    return round(float(np.mean(values)), 3) if values else 0.0


def confidence_profile(
    cluster: list[str],
    id_to_index: dict[str, int],
    similarity_matrix: np.ndarray | None,
    component_lookup: dict[tuple[int, int], dict[str, float]] | None,
) -> tuple[float, str]:
    """Calculate theme confidence and explain the supporting signal mix."""
    indices = [id_to_index[risk_id] for risk_id in cluster if risk_id in id_to_index]
    if len(indices) < 2 or similarity_matrix is None:
        return (
            MIN_THEME_CONFIDENCE,
            "Confidence is neutral because this is a single-risk theme with no internal pairwise similarity test.",
        )

    pair_scores = []
    semantic_scores = []
    metric_impact_scores = []
    taxonomy_scores = []
    driver_scores = []
    component_lookup = component_lookup or {}

    for left_position, left_index in enumerate(indices):
        for right_index in indices[left_position + 1 :]:
            pair_scores.append(float(similarity_matrix[left_index, right_index]))
            components = component_lookup.get((left_index, right_index), {})
            semantic_scores.append(float(components.get("Semantic_Similarity", 0.0)))
            metric_impact_scores.append(float(components.get("Metric_Impact_Similarity", 0.0)))
            taxonomy_scores.append(float(components.get("Taxonomy_Alignment_Score", 0.0)))
            driver_scores.append(float(components.get("Driver_Similarity", 0.0)))

    confidence = round(float(np.mean(pair_scores)), 3) if pair_scores else MIN_THEME_CONFIDENCE
    semantic_avg = float(np.mean(semantic_scores)) if semantic_scores else 0.0
    metric_impact_avg = float(np.mean(metric_impact_scores)) if metric_impact_scores else 0.0
    taxonomy_avg = float(np.mean(taxonomy_scores)) if taxonomy_scores else 0.0
    driver_avg = float(np.mean(driver_scores)) if driver_scores else 0.0
    strongest_signal = max(
        [
            ("risk title/description", semantic_avg * HYBRID_WEIGHTS["semantic"]),
            ("risk metric/impact comment", metric_impact_avg * HYBRID_WEIGHTS["metric_impact"]),
            ("taxonomy", taxonomy_avg * HYBRID_WEIGHTS["taxonomy"]),
            ("driver/root cause", driver_avg * HYBRID_WEIGHTS["driver"]),
        ],
        key=lambda item: item[1],
    )[0]
    rationale = (
        f"Confidence {confidence:.2f} is the average internal hybrid score "
        f"(risk title/description {semantic_avg:.2f}, metric/impact comment {metric_impact_avg:.2f}, "
        f"taxonomy {taxonomy_avg:.2f}, driver/root cause {driver_avg:.2f}); "
        f"largest weighted contribution is {strongest_signal}. "
        f"Themes below {MIN_THEME_CONFIDENCE:.2f} are excluded."
    )
    return confidence, rationale


def build_theme_records(
    data: pd.DataFrame,
    clusters: list[list[str]],
    relationships: pd.DataFrame,
    similarity_matrix: np.ndarray | None = None,
    component_lookup: dict[tuple[int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Create AI-enriched theme mapping records from clusters."""
    theme_records = []
    id_to_index = {risk_id: index for index, risk_id in enumerate(data["Group_ID"].astype(str).tolist())}
    for index, cluster in enumerate(clusters, start=1):
        theme_data = data[data["Group_ID"].astype(str).isin(cluster)].copy()
        dominant_l2 = dominant_value(theme_data["Taxonomy_L2"])
        dominant_root_cause_level = dominant_value(theme_data.apply(root_cause_level, axis=1))
        dominant_metric = dominant_value(theme_data["Risk_Metric"])
        dominant_method = dominant_value(theme_data["Assessment_Method"])
        dominant_division = dominant_value(theme_data["Business_Division"])
        dominant_l1 = dominant_value(theme_data["Taxonomy_L1"])
        shared_topic = common_topic(theme_data)
        label_source = dominant_l2 or dominant_l1 or dominant_root_cause_level
        theme_name = format_theme_label(label_source)
        root_fragment = title_fragment(dominant_root_cause_level)
        if root_fragment and root_fragment.lower() not in theme_name.lower():
            theme_name = f"{theme_name} - {root_fragment}"
        topic_fragment = title_fragment(shared_topic)
        base_label_text = f"{theme_name} {dominant_l1} {dominant_l2} {dominant_root_cause_level}"
        root_keywords = semantic_keywords(dominant_root_cause_level, limit=5)
        if topic_fragment and topic_should_split_theme(dominant_l2, root_keywords, topic_fragment, base_label_text):
            theme_name = f"{theme_name} - {topic_fragment}"
        taxonomies = sorted(
            {
                value
                for value in theme_data[["Taxonomy_L0", "Taxonomy_L1", "Taxonomy_L2"]]
                .astype(str)
                .stack()
                .tolist()
                if value and value.lower() != "nan"
            }
        )
        divisions = sorted(theme_data["Business_Division"].dropna().astype(str).unique().tolist())
        gcrs_values = unique_values(theme_data["GCRS"], limit=6)
        metric_values = unique_values(theme_data["Risk_Metric"], limit=6)
        method_values = unique_values(theme_data["Assessment_Method"], limit=6)
        metrics = unique_join(theme_data["Risk_Metric"], limit=4)
        methods = unique_join(theme_data["Assessment_Method"], limit=3)
        root_causes = unique_join(theme_data.apply(root_cause_level, axis=1), limit=3)
        root_evidence = unique_join(theme_data.apply(root_cause_evidence, axis=1), limit=3)
        taxonomy_l2_values = unique_join(theme_data["Taxonomy_L2"], limit=3)
        if similarity_matrix is not None:
            avg_similarity = average_internal_similarity_from_matrix(cluster, id_to_index, similarity_matrix)
        else:
            avg_similarity = average_internal_similarity(cluster, relationships)
        cross_business = len(divisions) > 1
        emerging = bool(cross_business and len(cluster) >= 2 and theme_data["Taxonomy_L1"].nunique() > 1)
        material_count, non_material_count = materiality_counts(theme_data)
        gap_text = theme_gap_text(theme_data)
        key_drivers = theme_key_drivers(theme_data, shared_topic)
        confidence_score, confidence_rationale = confidence_profile(
            cluster,
            id_to_index,
            similarity_matrix,
            component_lookup,
        )
        review_required, governance_check = theme_review_flag(theme_data, confidence_score)

        theme_records.append(
            {
                "Theme_ID": f"THM_{index:03d}",
                "Theme_Name": theme_name,
                "Primary_Driver": dominant_root_cause_level,
                "Primary_Taxonomy_L1": dominant_l1,
                "Primary_Taxonomy_L2": dominant_l2,
                "Primary_Metric": dominant_metric,
                "Primary_Method": dominant_method,
                "Primary_Division": dominant_division,
                "Common_Topic": shared_topic,
                "Key_Drivers": key_drivers,
                "Theme_Summary": (
                    f"- **Scope:** {len(theme_data)} risks related to {taxonomy_l2_values}.\n"
                    f"- **Common topic:** {shared_topic or 'No repeated title/description topic strong enough to use as a split key.'}\n"
                    f"- **Key drivers:** {', '.join(key_drivers) if key_drivers else root_causes}. "
                    f"Supporting evidence: {root_evidence or 'not provided'}.\n"
                    f"- **Coverage:** Business divisions include {', '.join(divisions) if divisions else 'none'}; "
                    f"Overall_Materiality count is {material_count} material / {non_material_count} non-material.\n"
                    f"- **Review note:** {governance_check}\n"
                    f"- **Measurement and governance:** Metrics include {metrics}; assessment methods include {methods}. "
                    f"{gap_text}"
                ),
                "Taxonomy_Alignment": taxonomies,
                "Business_Divisions": divisions,
                "GCRS_Values": gcrs_values,
                "Risk_Metrics": metric_values,
                "Assessment_Methods": method_values,
                "Risk_Count": int(len(theme_data)),
                "Material_Risks": material_count,
                "Non_Material_Risks": non_material_count,
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
                "Confidence_Score": confidence_score,
                "Confidence_Rationale": confidence_rationale,
                "Review_Required": review_required,
                "Governance_Check": governance_check,
                "Potential_Gap": gap_text,
            }
        )

    themes = pd.DataFrame(theme_records)
    if themes.empty:
        return themes

    themes = themes[
        (themes["Confidence_Score"] >= MIN_THEME_CONFIDENCE)
        & (themes["Risk_Count"] > 1)
        & (themes["Average_Similarity"] > 0)
    ].copy()
    if themes.empty:
        return themes

    themes = themes.reset_index(drop=True)
    themes["Theme_ID"] = [f"THM_{index:03d}" for index in range(1, len(themes) + 1)]
    themes["Embedding_Centroid_ID"] = [f"vec_{index:03d}" for index in range(1, len(themes) + 1)]
    return uniquify_theme_names(themes)


def differentiator_for_theme(row: pd.Series) -> str:
    """Choose a concise suffix when otherwise similar theme names remain."""
    root_cause = title_fragment(str(row.get("Primary_Driver") or ""))
    taxonomy_l2 = str(row.get("Primary_Taxonomy_L2") or "").strip()
    taxonomy_l1 = str(row.get("Primary_Taxonomy_L1") or "").strip()
    division = str(row.get("Primary_Division") or "").strip()

    if root_cause:
        return root_cause
    if taxonomy_l2:
        return taxonomy_l2
    if taxonomy_l1:
        return taxonomy_l1
    if division:
        return division
    return ""


def uniquify_theme_names(themes: pd.DataFrame) -> pd.DataFrame:
    """Ensure the review table does not show repeated suggested theme names."""
    if themes.empty or not themes["Theme_Name"].duplicated().any():
        return themes

    themes = themes.copy()
    for theme_name, rows in themes.groupby("Theme_Name"):
        if len(rows) == 1:
            continue

        used_suffixes: set[str] = set()
        for index, row in rows.iterrows():
            suffix = differentiator_for_theme(row)
            if not suffix or suffix in used_suffixes:
                suffix = f"Cluster {len(used_suffixes) + 1}"
            used_suffixes.add(suffix)
            themes.at[index, "Theme_Name"] = f"{theme_name} - {suffix}"

    return themes


def build_dimension_count_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create Business Division and GCRS count tables for ring charts."""
    if data.empty:
        return pd.DataFrame()

    rows = []
    for dimension, column in [("Business Division", "Business_Division"), ("GCRS", "GCRS")]:
        values = data[column].fillna("").astype(str).str.strip().replace("", "Unspecified")
        for value, count in values.value_counts().items():
            rows.append(
                {
                    "Dimension": dimension,
                    "Value": value,
                    "Risk_Count": int(count),
                }
            )

    return pd.DataFrame(rows).sort_values(["Dimension", "Risk_Count"], ascending=[True, False])


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
    embeddings = build_embedding_layer(scoped, payloads)
    hybrid_matrix, component_lookup = build_hybrid_similarity_matrix(scoped, embeddings)
    relationships = build_similarity_relationships(
        scoped,
        hybrid_matrix,
        top_k,
        relationship_threshold,
        component_lookup,
    )
    risk_ids = scoped["Group_ID"].astype(str).tolist()
    if target_theme_count is not None and target_theme_count < len(risk_ids):
        clusters = cluster_risks_to_target_by_similarity(risk_ids, hybrid_matrix, target_theme_count)
    else:
        clusters = cluster_risks(risk_ids, relationships, cluster_threshold)
    clusters = merge_duplicate_theme_clusters(scoped, clusters)
    themes = build_theme_records(scoped, clusters, relationships, hybrid_matrix, component_lookup)
    dimension_summary = build_dimension_count_summary(scoped)
    return themes, relationships, dimension_summary


def format_theme_list(values: object, fallback: str = "not available") -> str:
    """Format list-like theme evidence for narrative fallback text."""
    if isinstance(values, list):
        clean_values = [str(value).strip() for value in values if str(value).strip()]
        return ", ".join(clean_values) if clean_values else fallback
    clean = str(values or "").strip()
    return clean if clean else fallback


def build_theme_fallback_sections(themes: pd.DataFrame, relationships: pd.DataFrame) -> tuple[str, str]:
    """Create two deterministic theme-analysis cards when live GPT analysis is unavailable."""
    if themes.empty:
        message = "No theme candidates were generated for the current filtered inventory."
        return message, "No priority theme analysis is available until at least one theme passes the final criteria."

    cross_business = int(themes["Cross_Business"].sum())
    emerging = int(themes["Emerging_Indicator"].sum())
    top_themes = themes.sort_values(["Risk_Count", "Average_Similarity"], ascending=[False, False]).head(6)
    priority_themes = themes.sort_values(
        ["Material_Risks", "Risk_Count", "Average_Similarity"],
        ascending=[False, False, False],
    ).head(3)
    summary_lines = [
        f"The selected inventory scope is organized into **{len(themes)} suggested risk themes**. "
        f"**{cross_business}** themes span more than one business division, and **{emerging}** themes show cross-taxonomy patterns "
        "that may benefit from coordinated management attention.",
        "",
        "**Theme definitions, linkage, and evidence**",
    ]
    priority_lines = ["**Most important suggested themes**"]
    hotspot_lines = ["**Cross-division inconsistency hotspots**"]
    review_lines = ["**Human-review priorities**"]

    for row in top_themes.itertuples(index=False):
        divisions = format_theme_list(row.Business_Divisions)
        gcrs_values = format_theme_list(getattr(row, "GCRS_Values", ""))
        drivers = format_theme_list(row.Key_Drivers, row.Primary_Driver or "not available")
        risk_ids = format_theme_list(getattr(row, "Risk_IDs", [])[:4] if isinstance(getattr(row, "Risk_IDs", []), list) else getattr(row, "Risk_IDs", ""))
        summary_lines.append(
            f"- **{row.Theme_Name}** spans {divisions or 'one business division'}; "
            f"GCRS coverage includes {gcrs_values}; driver evidence is {drivers}. "
            f"Representative risk evidence includes {risk_ids}."
        )

    for row in priority_themes.itertuples(index=False):
        divisions = format_theme_list(row.Business_Divisions)
        gcrs_values = format_theme_list(getattr(row, "GCRS_Values", ""))
        drivers = format_theme_list(row.Key_Drivers, row.Primary_Driver or "not available")
        risk_ids = format_theme_list(getattr(row, "Risk_IDs", [])[:4] if isinstance(getattr(row, "Risk_IDs", []), list) else getattr(row, "Risk_IDs", ""))
        priority_lines.append(
            f"- **{row.Theme_Name}** should be reviewed first: {int(row.Risk_Count)} risks, "
            f"including {int(row.Material_Risks)} material risks. "
            f"Support: IDs {risk_ids}; businesses {divisions}; GCRS {gcrs_values}; driver evidence {drivers}. "
            "Transmission, exposure, and managed-action linkage should be validated where not explicit in the records."
        )

        metrics = format_theme_list(getattr(row, "Risk_Metrics", ""))
        methods = format_theme_list(getattr(row, "Assessment_Methods", ""))
        hotspot_lines.append(
            f"- **{row.Theme_Name}**: metrics include {metrics}; assessment methods include {methods}. "
            f"{row.Potential_Gap}"
        )
        review_lines.append(f"- **{row.Theme_Name}**: {row.Governance_Check}")

    theme_summary = "\n".join(summary_lines)
    theme_analysis = "\n".join(
        [
            *priority_lines,
            "",
            *hotspot_lines,
            "",
            *review_lines,
        ]
    )
    return theme_summary, theme_analysis


def build_theme_fallback_analysis(themes: pd.DataFrame, relationships: pd.DataFrame) -> str:
    """Create a deterministic narrative when live GPT analysis is unavailable."""
    theme_summary, theme_analysis = build_theme_fallback_sections(themes, relationships)
    return f"**Theme Summary**\n{theme_summary}\n\n**Theme Analysis**\n{theme_analysis}"
