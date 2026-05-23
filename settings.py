"""Shared constants for the risk taxonomy dashboard."""

from pathlib import Path


APP_TITLE = "Risk Taxonomy Dashboard"
DEFAULT_MODEL = "gpt-5.2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "inventory.JSON"

EXPECTED_COLUMNS = [
    "Group_ID",
    "Risk_Status",
    "Business_Division",
    "GCRS",
    "Legal_Entity",
    "SubLegal_Entity",
    "Risk_Type",
    "Taxonomy_L0",
    "Taxonomy_L1",
    "Taxonomy_L2",
    "Risk_Title",
    "Risk_Description",
    "Root_Cause_Driver",
    "Root_Cause",
    "Risk_Driver",
    "Cause_Driver",
    "Overall_Materiality",
    "Likelihood_Rating",
    "Likelihood_Comment",
    "Impact_Rating",
    "Impact_Numbers",
    "Risk_Metric",
    "Risk_Exposure",
    "Impact_Comment",
    "Assessment_Method",
    "Reporting_Quarter",
]

REQUIRED_COLUMNS = {
    "Taxonomy_L1",
    "Risk_Title",
    "Risk_Metric",
    "Assessment_Method",
}

SEARCH_COLUMNS = [
    "Group_ID",
    "Risk_Title",
    "Risk_Description",
    "Taxonomy_L0",
    "Taxonomy_L1",
    "Taxonomy_L2",
    "Root_Cause_Driver",
    "Root_Cause",
    "Risk_Driver",
    "Cause_Driver",
    "Risk_Metric",
    "Assessment_Method",
    "Business_Division",
    "GCRS",
]

AI_CONTEXT_COLUMNS = [
    "Group_ID",
    "Risk_Status",
    "Business_Division",
    "GCRS",
    "Legal_Entity",
    "SubLegal_Entity",
    "Risk_Type",
    "Taxonomy_L0",
    "Taxonomy_L1",
    "Taxonomy_L2",
    "Risk_Title",
    "Risk_Description",
    "Root_Cause_Driver",
    "Root_Cause",
    "Risk_Driver",
    "Cause_Driver",
    "Overall_Materiality",
    "Likelihood_Rating",
    "Risk_Metric",
    "Risk_Exposure",
    "Impact_Comment",
    "Assessment_Method",
    "Reporting_Quarter",
]

RISK_TABLE_COLUMNS = [
    "Group_ID",
    "Taxonomy_L0",
    "Taxonomy_L1",
    "Taxonomy_L2",
    "Risk_Title",
    "Risk_Status",
    "Risk_Type",
    "Business_Division",
    "Risk_Metric",
    "Assessment_Method",
    "Overall_Materiality",
    "Likelihood_Rating",
]
