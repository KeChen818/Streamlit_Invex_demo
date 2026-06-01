# Risk Taxonomy Grouping

Streamlit dashboard for exploring a flat risk inventory by `Taxonomy_L1`, comparing assessment methods and risk metrics, and asking inventory-specific questions through an AI chatbot.

## What It Does

- Summarizes risk counts by `Taxonomy_L1`.
- Shows risks under each Taxonomy L1 group through popover cards.
- Compares `Assessment_Method` and `Risk_Metric` using ring charts.
- Suggests higher-level AI Risk Themes using embedding-style similarity, theme clustering, GPT narrative analysis, `Overall_Materiality` counts, Business Division/GCRS ring charts, and a human review workflow.
- Generates AI narrative summaries for taxonomy patterns and method/metric differences.
- Provides an AI chatbot that answers questions about the current filtered inventory and filters the associated risk table.
- Excludes `Impact_Numbers` analysis by design. The app may use `Impact_Comment` text only to compare assumptions when asked by the prompt logic.

## Project Structure

```text
risk_taxonomy_dashboard/
  app.py                         # Thin Streamlit entrypoint
  README.md                      # Setup, run, and maintenance notes
  SPEC.md                        # Product and technical specification
  requirements.txt               # Python dependencies
  pages/
    2_AI_Risk_Theme_Classification.py
  .streamlit/
    config.toml                  # Streamlit theme and server config
  data/
    inventory.JSON               # Bundled private inventory file
  risk_dashboard/
    __init__.py
    bootstrap.py                 # Shared page loading and sidebar filter setup
    main.py                      # App orchestration
    settings.py                  # Constants, paths, schema columns
    styles.py                    # Page config and CSS
    data.py                      # JSON loading, normalization, filters
    analysis.py                  # Deterministic counts and fallback summaries
    ai.py                        # OpenAI Responses API and chatbot logic
    themes.py                    # Risk theme similarity, clustering, and review fields
    views.py                     # Streamlit page, tab, and component rendering
    utils.py                     # Formatting/parsing helpers
```

## Run Locally

```bash
cd "/Users/ke/Documents/New project/risk_taxonomy_dashboard"
pip install -r requirements.txt
streamlit run app.py
```

The app loads `data/inventory.JSON` automatically. There is no front-end data-source selector.

## Enable AI

AI summaries and chatbot answers use the OpenAI Responses API. The backend model defaults to `gpt-5.2` and can be overridden with `OPENAI_MODEL`; neither the model nor API status is shown in the front end.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-5.2"
streamlit run app.py
```

If `OPENAI_API_KEY` is not set or the API call fails, the app uses deterministic local fallback analysis.

## AI Risk Theme Classification

The `AI Risk Theme Classification` page adds an AI-enriched layer without changing official taxonomy fields.

- Builds a weighted semantic payload led by risk title and risk description, followed by risk metric/impact commentary, taxonomy, and driver/root-cause context.
- Extracts a repeated `Common_Topic` from risk titles and descriptions; one-off wording does not split a theme.
- Creates LangChain `Document` objects from the same weighted payload used by local embeddings, with risk ID, title, taxonomy, business division, and GCRS metadata for traceability.
- Uses `OpenAIEmbeddings` with `text-embedding-3-large` and a FAISS vector store when LangChain, FAISS, and `OPENAI_API_KEY` are available; otherwise it falls back to local hashed TF-IDF embeddings.
- Scores candidate relationships with a hybrid model: risk title/description semantic similarity 50%, risk metric/impact-comment similarity 25%, taxonomy alignment 15%, and driver/root-cause similarity 10%.
- Groups by common risk title and description semantics first, then metric/impact assumption evidence, taxonomy alignment, and driver/root-cause context.
- Excludes suggested themes with confidence below `0.50`, fewer than 2 risks, or zero average internal similarity, even when the target theme count is set higher.
- Writes the confidence basis into each theme summary and review row so the score is tied to title/description, metric/impact-comment, taxonomy, and driver evidence.
- Uses a `Target themes (max)` control so larger inventories can collapse into an executive-level range such as 20-30 themes for 300+ risks.
- Merges duplicate `Taxonomy_L1` / `Taxonomy_L2` / `Root_Cause_L0` / `Root_Cause_L1` / stable `Common_Topic` signatures so the review table does not repeat the same suggested theme name.
- Uses `Root_Cause_Comment` and `Early_Warning_Sign(EWS)` as supporting evidence in summaries instead of splitting themes.
- Uses top-K similarity search and configurable threshold bands:
  - `>0.90`: near duplicate
  - `0.82-0.90`: same theme
  - `0.72-0.82`: related
  - `<0.72`: weak relationship
- Clusters risks into suggested themes with deterministic target-count clustering over the hybrid similarity matrix, while still showing pairwise similar-risk relationships. Risk metric and assessment method remain supporting review fields, not primary grouping keys.
- Generates theme mapping records with `Theme_ID`, suggested theme name, summary, taxonomy alignment, business divisions, risk IDs, key drivers, confidence score, governance check, emerging indicator, and review fields.
- Supports reviewer actions: Accept, Rename, Merge, Split, Reject.
- Shows Business Division and GCRS count ring charts for the selected inventory scope.

`OPENAI_EMBEDDING_MODEL` can override the embedding model. The default is `text-embedding-3-large`. ChromaDB, Pinecone, or Azure AI Search can replace the FAISS layer later if enterprise metadata filtering or managed governance is needed.

## Input Schema

The dashboard expects a JSON list of risk records. Required columns:

- `Taxonomy_L1`
- `Risk_Title`
- `Risk_Metric`
- `Assessment_Method`

Important optional columns:

- `Group_ID`
- `Business_Division`
- `GCRS`
- `SubLegal_Entity`
- `Taxonomy_L0`
- `Taxonomy_L2`
- `Risk_Status`
- `Risk_Type`
- `Overall_Materiality`
- `Likelihood_Rating`
- `Risk_Description`
- `Root_Cause_L0`
- `Root_Cause_L1`
- `Root_Cause_Comment`
- `Early_Warning_Sign(EWS)`
- `Risk_Exposure`
- `Impact_Comment`
- `Reporting_Quarter`

Missing optional fields are filled with blanks. Missing `Group_ID` values are generated as `RISK-001`, `RISK-002`, and so on.

Sidebar filters are limited to `SubLegal_Entity`, `Risk Type`, `Taxonomy L0`, `Business Division`, `GCRS`, `Overall Materiality`, and a searchable `Group ID / Risk Title` lookup. `Risk Type` defaults to `Financial` when present.

## Development Notes

- Keep `app.py` thin. Add behavior in `risk_dashboard/` modules.
- Keep raw inventory data in `data/inventory.JSON`.
- Keep visual theme changes in `.streamlit/config.toml` and `risk_dashboard/styles.py`.
- Keep generated analytics count-based unless the spec explicitly changes the impact-analysis constraint.
- Keep official taxonomy columns authoritative; AI theme classification should only add suggested overlay fields.
- When editing AI prompts, preserve the instruction not to analyze or aggregate `Impact_Numbers`.

## Verification

Useful checks:

```bash
python -m compileall app.py risk_dashboard
python - <<'PY'
from risk_dashboard.data import load_json_records, normalize_data
from risk_dashboard.analysis import build_group_summary, build_method_metric
from risk_dashboard.themes import build_theme_classification

df = normalize_data(load_json_records())
themes, relationships, dimension_summary = build_theme_classification(df)
print(len(df), "records")
print(len(build_group_summary(df)), "Taxonomy L1 groups")
print(len(build_method_metric(df)), "method/metric rows")
print(len(themes), "suggested themes")
print(len(relationships), "similarity relationships")
print(len(dimension_summary), "dimension count rows")
PY
```
