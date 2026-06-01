"""Deterministic portfolio analytics and fallback narrative builders."""

import pandas as pd

from .utils import extract_assumption_basis, format_mapping, format_top_counts, top_counts, unique_join


def build_group_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filtered risks by Taxonomy L1 without impact analysis."""
    summary = (
        data.groupby("Taxonomy_L1", dropna=False)
        .agg(
            Risk_Count=("Group_ID", "nunique"),
            Material_Risks=("Is_Material", "sum"),
            Associated_SubLegal_Entity=("SubLegal_Entity", unique_join),
            Associated_Business_Division=("Business_Division", unique_join),
            Metrics=("Risk_Metric", unique_join),
            Methods=("Assessment_Method", unique_join),
            L2_Count=("Taxonomy_L2", lambda s: s.replace("", pd.NA).nunique()),
        )
        .reset_index()
    )

    sample_risks = (
        data.sort_values(["Taxonomy_L1", "Risk_Title"])
        .groupby("Taxonomy_L1")["Risk_Title"]
        .apply(lambda s: "; ".join(s.head(3)))
        .rename("Sample_Risks")
        .reset_index()
    )
    summary = summary.merge(sample_risks, on="Taxonomy_L1", how="left")
    return summary.sort_values(["Risk_Count", "Taxonomy_L1"], ascending=[False, True])


def build_method_metric(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate method/metric counts by Taxonomy L1 for ring charts and AI context."""
    comparison = (
        data.groupby(["Taxonomy_L1", "Assessment_Method", "Risk_Metric"], dropna=False)
        .agg(
            Risk_Count=("Group_ID", "nunique"),
            Material_Risks=("Is_Material", "sum"),
        )
        .reset_index()
    )
    return comparison


def build_taxonomy_fallback_summary(data: pd.DataFrame, group_summary: pd.DataFrame) -> str:
    """Build a deterministic Taxonomy L1 summary when GPT is unavailable."""
    total_groups = data["Taxonomy_L1"].nunique()
    total_risks = len(data)
    largest_groups = format_top_counts(top_counts(data["Taxonomy_L1"], limit=3))

    division_lines = []
    for division, subset in data.groupby("Business_Division", dropna=False):
        division_name = str(division or "Unassigned")
        dominant_groups = format_top_counts(top_counts(subset["Taxonomy_L1"], limit=2))
        dominant_metrics = format_top_counts(top_counts(subset["Risk_Metric"], limit=2))
        dominant_methods = format_top_counts(top_counts(subset["Assessment_Method"], limit=2))
        division_lines.append(
            f"- **{division_name}**: {len(subset)} risks; dominant L1 groups: {dominant_groups}; "
            f"dominant metrics: {dominant_metrics}; dominant methods: {dominant_methods}."
        )

    gap_lines = []
    all_divisions = set(data["Business_Division"].dropna().astype(str).str.strip())
    for group_name in group_summary["Taxonomy_L1"].tolist():
        subset = data[data["Taxonomy_L1"] == group_name]
        divisions = sorted(subset["Business_Division"].dropna().astype(str).unique().tolist())
        metric_by_division = {
            division: unique_join(part["Risk_Metric"], limit=3)
            for division, part in subset.groupby("Business_Division", dropna=False)
        }
        method_by_division = {
            division: unique_join(part["Assessment_Method"], limit=3)
            for division, part in subset.groupby("Business_Division", dropna=False)
        }
        metric_sets = {value for value in metric_by_division.values()}
        method_sets = {value for value in method_by_division.values()}

        if len(divisions) > 1 and (len(metric_sets) > 1 or len(method_sets) > 1):
            gap_lines.append(
                f"- **{group_name}** appears in {len(divisions)} divisions with inconsistent patterns. "
                f"Metrics by division: {format_mapping(metric_by_division)}. "
                f"Methods by division: {format_mapping(method_by_division)}."
            )
        elif len(divisions) == 1 and len(all_divisions) > 1:
            gap_lines.append(
                f"- **{group_name}** is only observed in **{divisions[0]}**; validate whether other divisions "
                "do not carry this exposure or whether the taxonomy mapping is incomplete."
            )

    if not gap_lines:
        gap_lines.append("- No obvious method or metric coverage gap is visible in the current filtered data.")

    return "\n".join(
        [
            f"**Portfolio shape:** {total_risks} risks are mapped into **{total_groups} Taxonomy L1 groups**. "
            f"Largest groups by count: {largest_groups}.",
            "",
            "**Dominant patterns by business division**",
            *division_lines,
            "",
            "**Potential metric or method gaps across business divisions**",
            *gap_lines[:8],
        ]
    )


def build_compare_fallback_analysis(data: pd.DataFrame) -> str:
    """Build deterministic method/metric comparison text when GPT is unavailable."""
    similarity_lines = []
    for group_name, subset in data.groupby("Taxonomy_L1", dropna=False):
        group_ids = unique_join(subset["Group_ID"], limit=5)
        divisions = unique_join(subset["Business_Division"], limit=4)
        gcrs = unique_join(subset["GCRS"], limit=4)
        method_mix = format_top_counts(top_counts(subset["Assessment_Method"], limit=4))
        metric_mix = format_top_counts(top_counts(subset["Risk_Metric"], limit=4))
        method_count = subset["Assessment_Method"].nunique()
        metric_count = subset["Risk_Metric"].nunique()

        if method_count == 1 and metric_count == 1:
            similarity = "high similarity"
        elif method_count == 1 or metric_count == 1:
            similarity = "partial similarity"
        else:
            similarity = "high difference"

        similarity_lines.append(
            f"- **{group_name}** shows **{similarity}** across Group IDs {group_ids}. "
            f"Business divisions: {divisions}; GCRS: {gcrs}. "
            f"Method mix: {method_mix}. Metric mix: {metric_mix}."
        )

    assumption_lines = []
    for (group_name, method, metric), subset in data.groupby(
        ["Taxonomy_L1", "Assessment_Method", "Risk_Metric"],
        dropna=False,
    ):
        if len(subset) < 2:
            continue

        comparison = []
        bases = set()
        for risk in subset.sort_values("Group_ID").to_dict("records"):
            basis = extract_assumption_basis(risk.get("Impact_Comment", ""))
            bases.add(basis)
            comparison.append(
                f"{risk['Group_ID']} ({risk['Business_Division']} / {risk['GCRS']}): {basis}"
            )

        if len(bases) > 1:
            assumption_lines.append(
                f"- **{group_name}** with **{method}** and **{metric}** uses different assumption bases: "
                + "; ".join(comparison)
                + "."
            )

    if not assumption_lines:
        assumption_lines.append(
            "- No same-group, same-method, same-metric records with different impact-commentary assumptions "
            "were found in the current filtered data."
        )

    return "\n".join(
        [
            "**1. Method and metric similarity across Business Division, GCRS, and Group ID**",
            *similarity_lines,
            "",
            "**2. Assumption differences in impact commentary for comparable risks**",
            *assumption_lines[:8],
        ]
    )
