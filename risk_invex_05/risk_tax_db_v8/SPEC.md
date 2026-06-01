# Risk Taxonomy Grouping Specification

## Purpose

Risk Taxonomy Grouping helps users review a flat risk inventory by Taxonomy L1 group, compare how risk methods and metrics are used, classify suggested AI Risk Themes, and ask natural-language questions about the filtered inventory.

The application is an internal analytical template, not a regulatory reporting system. It prioritizes explainable grouping, count-based comparison, and transparent links back to the underlying risk records.

## Users

- Risk inventory owners reviewing taxonomy coverage.
- Risk method and metric reviewers comparing consistency across business divisions.
- Theme reviewers validating suggested cross-business risk groupings.
- Analysts who need quick answers about which records match a risk topic, method, metric, division, or GCRS.

## Data Source

The app loads a bundled JSON inventory from:

```text
data/inventory.JSON
```

The front end must not expose a JSON path input, upload control, model selector, API key status, or source-path label.

## Required Data Fields

Required:

- `Taxonomy_L1`
- `Risk_Title`
- `Risk_Metric`
- `Assessment_Method`

Recommended:

- `Group_ID`
- `Risk_Status`
- `Business_Division`
- `GCRS`
- `Legal_Entity`
- `SubLegal_Entity`
- `Risk_Type`
- `Taxonomy_L0`
- `Taxonomy_L2`
- `Risk_Description`
- `Root_Cause_L0`
- `Root_Cause_L1`
- `Root_Cause_Comment`
- `Early_Warning_Sign(EWS)`
- `Overall_Materiality`
- `Likelihood_Rating`
- `Risk_Exposure`
- `Impact_Comment`
- `Reporting_Quarter`

## Functional Requirements

### 1. Filters

The sidebar provides slicers for:

- SubLegal entity
- Risk type
- Taxonomy L0
- Business division
- GCRS
- Overall materiality
- Group ID / Risk title lookup

`Risk Type` defaults to `Financial` when that value is available in the inventory.

All dashboard tabs and pages operate on the currently filtered data.

### 2. Taxonomy L1 Groups

The tab must show:

- Total count metrics at the top of the page.
- A bar chart of risk count by `Taxonomy_L1`.
- A summary table with risk count, material risk count, Taxonomy L2 count, metrics, and methods.
- AI Taxonomy Summary narrative.
- Card view for each `Taxonomy_L1`.
- A popover on each card listing the risks under that group.

The AI Taxonomy Summary must address:

- How many Taxonomy L1 groups exist.
- Largest groups by count.
- Dominant Taxonomy L1 groups and risk metrics by `Business_Division`.
- Potential `Risk_Metric` or `Assessment_Method` gaps across business divisions.

### 3. Method and Metric Compare

The tab must show:

- Ring chart for assessment method mix.
- Ring chart for risk metric mix.
- Taxonomy L1 multiselect controlling the chart and AI analysis scope.
- AI Method and Metric Analysis narrative.

The AI analysis must answer:

- How similar or different each Taxonomy L1 group is when comparing `Assessment_Method` and `Risk_Metric` across `Business_Division`, `GCRS`, and `Group_ID`.
- For same group, same method, and similar or same metric, what assumption differences exist in `Impact_Comment`.

### 4. AI Risk Theme Classification

The dedicated page must add a suggested theme layer while preserving official taxonomy authority.

The AI layer must not overwrite:

- `Taxonomy_L1`
- `Taxonomy_L2`
- Regulatory classification fields

The AI layer may add:

- Suggested Risk Theme
- Similar Risk Relationships
- Emerging Clusters
- Cross-business linkage
- Key drivers
- Confidence score
- Governance review flag

The page must show:

- Controls for target theme count maximum, top-K similar risks, and relationship threshold.
- Count metrics for suggested themes, similar relationships, material risks, and non-material risks.
- AI Theme Analysis narrative.
- Bar chart of suggested theme counts stacked by direct `Overall_Materiality` count; no AI materiality classification is applied.
- Business Division and GCRS count ring charts.
- Human Review Workflow table.
- Theme detail expanders with structured summary bullets and mapped risks.
- Similar Risk Relationships table.

Theme grouping must use a hybrid score led by title/description semantics instead of taxonomy alone:

- Risk title and risk description semantic similarity: 50%
- Risk metric and impact-comment similarity: 25%
- Taxonomy alignment: 15%
- Risk driver/root-cause similarity: 10%

Suggested themes with confidence below `0.50`, fewer than 2 risks, or zero average internal similarity must be excluded from the AI summary, charts, human review workflow, and detail cards. Theme summaries must include the confidence rationale so the score is connected to the evidence used for grouping.

When available, the embedding layer uses LangChain `OpenAIEmbeddings` with `text-embedding-3-large` and a FAISS vector store. If those dependencies or `OPENAI_API_KEY` are unavailable, the app must fall back to local deterministic embeddings.

Theme candidate threshold bands:

- `>0.90`: near duplicate
- `0.82-0.90`: same theme
- `0.72-0.82`: related
- `<0.72`: weak relationship

Human reviewer actions:

- Accept
- Rename
- Merge
- Split
- Reject

Reviewer edits are maintained in Streamlit session state for the current browser session.

#### Theme Classification MVP Pipeline

```text
Risk Inventory Source
        ↓
Data Standardization
        ↓
Weighted Semantic Payload Construction
        ↓
Local Embedding Generation
        ↓
Similarity Search
        ↓
Target-Count Theme Clustering
        ↓
GPT Theme Narrative
        ↓
Human Review Workflow
        ↓
Streamlit Dashboard
```

#### Semantic Payload

Each risk is represented with:

```text
Risk ID
Risk Name, highest-weight grouping signal
Risk Description
Risk Metric / Impact Comment
Taxonomy L1 / Taxonomy L2
Root Cause Driver
```

For 300+ risk records, the page should default to an executive-level target near 25 suggested themes. Users may adjust the target count when they need tighter or broader grouping.

After target-count clustering, duplicate `Taxonomy_L1` / `Taxonomy_L2` / `Root_Cause_L0` / `Root_Cause_L1` / stable `Common_Topic` signatures should be merged and any remaining duplicate labels should receive concise differentiators. `Common_Topic` should be derived from repeated title/description wording across multiple records, not from one individual title. `Root_Cause_Comment` and `Early_Warning_Sign(EWS)` should enrich summaries instead of splitting themes. `Risk_Metric` and `Impact_Comment` should support the secondary grouping score, while `Assessment_Method` remains a supporting review field.

#### Theme Mapping Output

The app must produce theme records with:

- `Theme_ID`
- `Theme_Name`
- `Theme_Summary`
- `Common_Topic`
- `Taxonomy_Alignment`
- `Business_Divisions`
- `Risk_Count`
- `Risk_IDs`
- `Embedding_Centroid_ID`
- `LLM_Generated`
- `Human_Reviewed`
- `Review_Action`
- `Reviewed_By`
- `Review_Date`
- `Created_Quarter`
- `Emerging_Indicator`

#### Future Vector Store Path

The MVP uses local hashed TF-IDF embeddings and in-memory similarity search. Future deployments may replace this layer with:

- OpenAI `text-embedding-3-large` or `text-embedding-3-small`
- Local `BAAI/bge-large-en-v1.5` or `all-MiniLM-L6-v2`
- FAISS for local vector search
- ChromaDB for metadata filtering
- Pinecone or Azure AI Search for enterprise vector search

Recommended future clustering options:

- HDBSCAN for irregular clusters
- Agglomerative clustering for explainability
- KMeans for simple fixed-count MVP grouping

### 5. AI Chatbot

The Risk Register tab is replaced by an AI Chatbot tab.

The chatbot must:

- Answer questions using only the currently filtered inventory.
- Return associated `Group_ID` values.
- Filter the associated risk table to those IDs.
- Fall back to deterministic local matching if GPT is unavailable.

The associated-risk table must include:

- Group ID
- Taxonomy fields
- Risk title
- Status
- Type
- Business division
- Metric
- Method
- Materiality
- Likelihood rating

## AI Requirements

- Use the OpenAI Responses API.
- Default backend model: `gpt-5.2`.
- Allow backend override through `OPENAI_MODEL`.
- Do not show the model name, API key status, or data source in the front end.
- If `OPENAI_API_KEY` is absent or a call fails, use deterministic local fallback logic.
- Prompts must instruct the model not to invent records, counts, IDs, metrics, methods, divisions, GCRS values, or assumptions.
- Theme analysis prompts must state that official taxonomy remains authoritative and AI themes are suggestions only.
- Chatbot GPT responses must return structured JSON with:
  - `answer_markdown`
  - `matching_group_ids`

## Impact Analysis Constraint

The app must not analyze, aggregate, compare, or chart `Impact_Numbers`.

Allowed:

- `Impact_Comment` may be used as text evidence to compare assumptions for otherwise comparable risks.

Not allowed:

- Summing impact.
- Averaging impact.
- Ranking by impact.
- Displaying impact values in summaries or tables.
- Using impact values as chart measures.

## UX and Theme Requirements

Theme:

- Background: `#f5f2ec`
- Card background: `#fff`
- Ink: `#1a1814`
- Soft ink: `#5c5650`
- Line: `#e3ddd2`
- Soft line: `#ecebe6`
- Accent: `#d64545`

Streamlit theme config must live in:

```text
.streamlit/config.toml
```

Custom page and component CSS must live in:

```text
risk_dashboard/styles.py
```

## Architecture

The app should remain modular:

- `app.py`: entrypoint only.
- `risk_dashboard/main.py`: orchestration.
- `risk_dashboard/settings.py`: constants and schema.
- `risk_dashboard/styles.py`: theme and CSS.
- `risk_dashboard/data.py`: load, normalize, filter.
- `risk_dashboard/analysis.py`: deterministic analytics and fallback narratives.
- `risk_dashboard/ai.py`: OpenAI calls and chatbot logic.
- `risk_dashboard/bootstrap.py`: shared inventory load and sidebar filter setup.
- `risk_dashboard/themes.py`: semantic payloads, optional OpenAI/FAISS embeddings, hybrid similarity relationships, theme clustering, Business Division/GCRS count summaries, and review fields.
- `risk_dashboard/views.py`: Streamlit rendering.
- `risk_dashboard/utils.py`: small shared helpers.

## Acceptance Criteria

- `streamlit run app.py` starts the app.
- The app loads `data/inventory.JSON` without front-end data-source controls.
- The front end does not display model name, API key status, or file source.
- Taxonomy L1 summary works from filtered data.
- Method and Metric Compare uses ring charts.
- AI Theme Classification produces suggested themes, similarity relationships, Business Division/GCRS count charts, and reviewer actions.
- AI Chatbot filters the associated-risk table by returned or fallback `Group_ID`s.
- No dashboard page analyzes `Impact_Numbers`.
- Python files compile successfully.
