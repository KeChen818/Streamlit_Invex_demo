"""Deterministic portfolio analytics used by dashboard views."""

import pandas as pd

from .utils import unique_join


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
