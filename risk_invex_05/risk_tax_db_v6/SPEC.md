# Risk Taxonomy Dashboard Specification

## Purpose

The Risk Taxonomy Dashboard helps users review a flat risk inventory by Taxonomy L1 group, compare how risk methods and metrics are used, and ask natural-language questions about the filtered inventory.

The application is an internal analytical template, not a regulatory reporting system. It prioritizes explainable grouping, count-based comparison, and transparent links back to the underlying risk records.

## Users

- Risk inventory owners reviewing taxonomy coverage.
- Risk method and metric reviewers comparing consistency across business divisions.
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
- `Overall_Materiality`
- `Likelihood_Rating`
- `Risk_Exposure`
- `Impact_Comment`
- `Reporting_Quarter`

## Functional Requirements

### 1. Filters

The sidebar provides slicers for:

- Reporting quarter
- Taxonomy L0
- Taxonomy L1 group
- Business division
- Risk type
- Materiality
- Status
- Search

All dashboard tabs operate on the currently filtered data.

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

### 4. AI Chatbot

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
- `risk_dashboard/views.py`: Streamlit rendering.
- `risk_dashboard/utils.py`: small shared helpers.

## Acceptance Criteria

- `streamlit run app.py` starts the app.
- The app loads `data/inventory.JSON` without front-end data-source controls.
- The front end does not display model name, API key status, or file source.
- Taxonomy L1 summary works from filtered data.
- Method and Metric Compare uses ring charts.
- AI Chatbot filters the associated-risk table by returned or fallback `Group_ID`s.
- No dashboard page analyzes `Impact_Numbers`.
- Python files compile successfully.
