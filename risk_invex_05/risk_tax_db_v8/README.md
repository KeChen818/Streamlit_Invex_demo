# Risk Taxonomy Dashboard

Streamlit dashboard for exploring a flat risk inventory by `Taxonomy_L1`, comparing assessment methods and risk metrics, and asking inventory-specific questions through an AI chatbot.

## What It Does

- Summarizes risk counts by `Taxonomy_L1`.
- Shows risks under each Taxonomy L1 group through popover cards.
- Compares `Assessment_Method` and `Risk_Metric` using ring charts.
- Suggests higher-level AI Risk Themes using embedding-style similarity, theme clustering, GPT narrative analysis, QoQ counts, and a human review workflow.
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

- Builds a semantic payload from risk ID, title, description, exposure, business division, taxonomy, metric, and method.
- Creates local hashed TF-IDF embeddings for MVP similarity.
- Uses top-K similarity search and configurable threshold bands:
  - `>0.90`: near duplicate
  - `0.82-0.90`: same theme
  - `0.72-0.82`: related
  - `<0.72`: weak relationship
- Clusters risks into suggested themes through connected similarity pairs.
- Generates theme mapping records with `Theme_ID`, suggested theme name, summary, taxonomy alignment, business divisions, risk IDs, emerging indicator, and review fields.
- Supports reviewer actions: Accept, Rename, Merge, Split, Reject.
- Shows QoQ theme counts when multiple reporting quarters exist.

Cloud embeddings such as `text-embedding-3-large` or `text-embedding-3-small`, and vector stores such as FAISS, ChromaDB, Pinecone, or Azure AI Search can replace the local MVP embedding/search layer later. The current app keeps that implementation local so it runs from the bundled inventory without extra infrastructure.

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
- `Taxonomy_L0`
- `Taxonomy_L2`
- `Risk_Status`
- `Risk_Type`
- `Overall_Materiality`
- `Likelihood_Rating`
- `Risk_Description`
- `Risk_Exposure`
- `Impact_Comment`
- `Reporting_Quarter`

Missing optional fields are filled with blanks. Missing `Group_ID` values are generated as `RISK-001`, `RISK-002`, and so on.

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
themes, relationships, quarter_summary = build_theme_classification(df)
print(len(df), "records")
print(len(build_group_summary(df)), "Taxonomy L1 groups")
print(len(build_method_metric(df)), "method/metric rows")
print(len(themes), "suggested themes")
print(len(relationships), "similarity relationships")
PY
```
