# Risk Taxonomy Grouping - Intern Developer Guide

This project is a Streamlit dashboard for exploring a flat risk inventory, comparing taxonomy/method/metric patterns, generating suggested AI risk themes, and asking inventory-specific questions through an AI chatbot.

Read this file first. It explains how the Python files relate to each other, where each responsibility lives, how data moves through the app, and how to add future stages without breaking the current structure.

## 1. Quick Start

```bash
cd "/Users/ke/Documents/New project 3/risk_taxonomy_dashboard_Intern"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-5.2"
streamlit run app.py
```

The app always loads:

```text
data/inventory.JSON
```

The front end must not expose a file picker, JSON path input, model selector, API key status, or source-path label.

## 2. Product Summary

The dashboard has two Streamlit pages:

1. `Risk Taxonomy Grouping`
   - Top-level risk count metrics.
   - `Taxonomy_L1` group summary.
   - Method and metric comparison charts.
   - GPT-generated taxonomy and method/metric analysis.
   - AI chatbot with an associated-risk table.

2. `AI Risk Theme Classification`
   - Suggested AI risk themes.
   - OpenAI embeddings plus FAISS similarity search.
   - Hybrid similarity scoring.
   - Target-count clustering.
   - GPT-generated theme summary and analysis.
   - Human review workflow.
   - Similar-risk relationship evidence table.

`Impact_Numbers` must not be analyzed, aggregated, ranked, averaged, or charted. `Impact_Comment` may be used as text evidence when comparing assumptions.

## 3. Directory Map

```text
risk_taxonomy_dashboard_Intern/
  app.py
  README.md
  SPEC.md
  requirements.txt
  pages/
    2_AI_Risk_Theme_Classification.py
  .streamlit/
    config.toml
  data/
    inventory.JSON
  risk_dashboard/
    __init__.py
    ai.py
    analysis.py
    bootstrap.py
    data.py
    main.py
    settings.py
    styles.py
    themes.py
    utils.py
    views.py
```

The structure is intentionally simple:

- `app.py` chooses which Streamlit page to run.
- `pages/` contains the dedicated second page.
- `risk_dashboard/` contains the reusable app code.
- `data/` contains the bundled inventory.
- `settings.py` centralizes constants and schema lists.
- `views.py` owns rendering.
- `ai.py` owns GPT calls.
- `themes.py` owns theme classification logic.

## 4. High-Level Python Relationship

Main page flow:

```text
app.py
  -> risk_dashboard.main.main()
     -> styles.configure_page()
     -> styles.inject_css()
     -> ai.selected_model()
     -> bootstrap.load_filtered_inventory()
        -> data.load_json_records()
        -> data.normalize_data()
        -> data.filter_data()
     -> analysis.build_group_summary()
     -> analysis.build_method_metric()
     -> views.metric_cards()
     -> views.render_group_summary()
        -> ai.get_taxonomy_ai_summary()
           -> ai.request_openai_text()
     -> views.render_comparison()
        -> ai.get_compare_ai_analysis()
           -> ai.request_openai_text()
     -> views.render_ai_chatbot()
        -> ai.ask_inventory_chatbot()
           -> ai.request_openai_text()
```

AI Risk Theme Classification page flow:

```text
app.py
  -> pages/2_AI_Risk_Theme_Classification.py
     -> styles.configure_page()
     -> styles.inject_css()
     -> ai.selected_model()
     -> bootstrap.load_filtered_inventory()
     -> views.metric_cards()
     -> views.render_theme_classification()
        -> themes.build_theme_classification()
           -> themes.build_semantic_payload()
           -> themes.build_embedding_layer()
              -> themes.build_openai_faiss_embeddings()
                 -> langchain_openai.OpenAIEmbeddings
                 -> langchain_community.vectorstores.FAISS
           -> themes.build_hybrid_similarity_matrix()
           -> themes.build_similarity_relationships()
           -> themes.cluster_risks_to_target_by_similarity()
           -> themes.merge_duplicate_theme_clusters()
           -> themes.build_theme_records()
           -> themes.build_dimension_count_summary()
        -> ai.get_theme_ai_analysis()
           -> ai.request_openai_text()
        -> views.render_theme_detail_cards()
        -> views.render_theme_review_editor()
        -> views.render_theme_review_controls()
```

The main idea: data is loaded once per page, normalized once, filtered in the sidebar, then passed into analytics, AI, and rendering functions.

## 5. File-by-File Purpose And Relationships

### `app.py`

Purpose:

- Streamlit entrypoint.
- Defines the app navigation.
- Keeps business logic out of the root file.

Important objects:

- `AI_THEME_PAGE`: path to `pages/2_AI_Risk_Theme_Classification.py`.
- `main()`: creates two `st.Page` objects and runs the selected page.

Imports:

- `risk_dashboard.main.main` as the main dashboard renderer.
- `risk_dashboard.settings.APP_TITLE`.

Called by:

- `streamlit run app.py`.

Future development:

- Add a new page here only when the app needs a new top-level Streamlit route.
- Do not add charts, data loading, or AI calls here.

### `pages/2_AI_Risk_Theme_Classification.py`

Purpose:

- Dedicated Streamlit page for AI theme classification.
- Allows the theme page to be opened directly through Streamlit navigation.

Important objects:

- `PROJECT_ROOT`: adds the project root to `sys.path` so the page can import `risk_dashboard`.
- `main()`: configures the page, loads filtered data, renders metrics, and calls `render_theme_classification()`.

Imports:

- `risk_dashboard.ai.selected_model`
- `risk_dashboard.bootstrap.load_filtered_inventory`
- `risk_dashboard.styles.configure_page`
- `risk_dashboard.styles.inject_css`
- `risk_dashboard.views.metric_cards`
- `risk_dashboard.views.render_theme_classification`

Called by:

- `app.py` through `st.Page(AI_THEME_PAGE, ...)`.

Future development:

- Keep this file thin.
- If the theme page needs new controls or components, add those to `views.py`.
- If the theme algorithm changes, add that to `themes.py`.

### `risk_dashboard/__init__.py`

Purpose:

- Marks `risk_dashboard` as a Python package.
- Keeps imports stable across app pages.

Current content:

- Package docstring only.

Future development:

- Usually leave this file alone.
- Do not put app initialization or shared state here.

### `risk_dashboard/main.py`

Purpose:

- Orchestrates the main dashboard page.
- Connects data loading, analytics, Streamlit tabs, and download controls.

Important flow:

1. Configure the page and inject CSS.
2. Read selected OpenAI model.
3. Load and filter inventory.
4. Build deterministic summary tables.
5. Render metric cards.
6. Render three tabs:
   - `Taxonomy L1 Groups`
   - `Method and Metric Compare`
   - `AI Chatbot`
7. Add sidebar CSV download for the group summary.

Imports:

- `ai.selected_model`
- `analysis.build_group_summary`
- `analysis.build_method_metric`
- `bootstrap.load_filtered_inventory`
- `settings.APP_TITLE`
- `styles.configure_page`
- `styles.inject_css`
- `views.metric_cards`
- `views.render_ai_chatbot`
- `views.render_comparison`
- `views.render_group_summary`

Called by:

- `app.py`.

Future development:

- Add a new main-dashboard tab here if it belongs on the first page.
- Keep tab rendering delegated to `views.py`.
- Keep calculations delegated to `analysis.py`, `themes.py`, or a new domain module.

### `risk_dashboard/settings.py`

Purpose:

- Centralizes constants, paths, and schema definitions.
- Prevents schema lists from being duplicated across modules.

Important constants:

- `APP_TITLE`: title used by the Streamlit app.
- `DEFAULT_MODEL`: default GPT model, currently `gpt-5.2`.
- `CHAT_ASSOCIATED_RISK_LIMIT`: max rows/IDs used by chatbot-related views.
- `PROJECT_ROOT`: project directory.
- `DATA_PATH`: bundled JSON inventory path.
- `EXPECTED_COLUMNS`: full normalized inventory column order.
- `REQUIRED_COLUMNS`: required input fields.
- `SEARCH_COLUMNS`: currently available search-oriented fields.
- `AI_CONTEXT_COLUMNS`: fields passed into GPT prompts.
- `RISK_TABLE_COLUMNS`: default associated-risk table columns.

Used by:

- `ai.py`
- `data.py`
- `main.py`
- `styles.py`
- `views.py`

Future development:

- Add new inventory columns here first.
- If a new field should appear in GPT context, add it to `AI_CONTEXT_COLUMNS`.
- If a new field should appear in the chatbot table, add it to `RISK_TABLE_COLUMNS`.
- Avoid hard-coding column names repeatedly in views or AI prompts when a shared list belongs here.

### `risk_dashboard/styles.py`

Purpose:

- Owns Streamlit page configuration and custom CSS.
- Keeps visual styling separate from app logic.

Important functions:

- `configure_page(page_title=None)`: calls `st.set_page_config()` with wide layout and sidebar expanded.
- `inject_css()`: injects CSS variables, card styles, table styles, score bars, badges, and review table styling.

Imports:

- `settings.APP_TITLE`

Used by:

- `risk_dashboard/main.py`
- `pages/2_AI_Risk_Theme_Classification.py`

Future development:

- Add visual changes here or in `.streamlit/config.toml`.
- Do not place business rules or data transformations in CSS helpers.
- If a view needs a new CSS class, define the class here and apply it in `views.py`.

### `risk_dashboard/bootstrap.py`

Purpose:

- Shared page bootstrapping for inventory loading and sidebar filters.
- Ensures both Streamlit pages use the same filtered data behavior.

Important function:

- `load_filtered_inventory()`
  - Loads JSON through `data.load_json_records()`.
  - Stops the page with `st.error()` if the JSON cannot load.
  - Stops the page if no records exist.
  - Normalizes records through `data.normalize_data()`.
  - Renders shared sidebar filters through `data.filter_data()`.
  - Stops the page if filters produce no records.
  - Returns the filtered dataframe.

Imports:

- `data.filter_data`
- `data.load_json_records`
- `data.normalize_data`

Used by:

- `risk_dashboard/main.py`
- `pages/2_AI_Risk_Theme_Classification.py`

Future development:

- Add app-wide data loading behavior here.
- Add page-specific controls in `views.py`, not here.
- Keep all pages using this helper unless a future page intentionally needs a different scope.

### `risk_dashboard/data.py`

Purpose:

- Loads the bundled JSON.
- Normalizes input records into a predictable dataframe.
- Applies shared sidebar filters.

Important functions:

- `records_from_payload(payload)`
  - Accepts common JSON shapes:
    - list of records
    - dict with `records`, `risks`, `data`, or `items`
    - single dict record
  - Returns only dictionary records.

- `load_json_records(data_path=DATA_PATH)`
  - Opens `data/inventory.JSON`.
  - Parses JSON.
  - Returns normalized record dictionaries through `records_from_payload()`.

- `normalize_data(records)`
  - Builds a pandas dataframe.
  - Validates `REQUIRED_COLUMNS`.
  - Adds missing `EXPECTED_COLUMNS` as blanks.
  - Strips text fields.
  - Generates missing `Group_ID` values as `RISK-001`, `RISK-002`, etc.
  - Replaces blank `Taxonomy_L1` with `Unmapped`.
  - Replaces blank `Risk_Metric` and `Assessment_Method` with `Unspecified`.
  - Creates `Likelihood_Score` from `Likelihood_Rating`.
  - Creates `Is_Material` from `Overall_Materiality`.

- `sorted_options(data, column)`
  - Builds stable filter choices.

- `risk_lookup_options(data)`
  - Builds searchable `Group_ID | Risk_Title` labels.

- `filter_data(data)`
  - Renders sidebar filters:
    - `SubLegal_Entity`
    - `Risk Type`
    - `Taxonomy L0`
    - `Business Division`
    - `GCRS`
    - `Overall Materiality`
    - `Group ID / Risk Title`
  - Defaults `Risk Type` to `Financial` when present.
  - Returns filtered dataframe.

Imports:

- `settings.DATA_PATH`
- `settings.EXPECTED_COLUMNS`
- `settings.REQUIRED_COLUMNS`

Used by:

- `bootstrap.py`
- Verification scripts in this README.

Future development:

- Add new source columns to `EXPECTED_COLUMNS` in `settings.py`.
- Add new normalization rules here.
- Add new global sidebar filters here only if every page should use them.
- Keep page-specific filters in `views.py`.

### `risk_dashboard/analysis.py`

Purpose:

- Builds deterministic count tables used by views and GPT prompts.
- Does not generate narrative AI text.

Important functions:

- `build_group_summary(data)`
  - Groups by `Taxonomy_L1`.
  - Computes:
    - risk count
    - material risk count
    - associated SubLegal entities
    - associated business divisions
    - metrics
    - methods
    - Taxonomy L2 count
    - sample risk titles
  - Sorts largest groups first.

- `build_method_metric(data)`
  - Groups by `Taxonomy_L1`, `Assessment_Method`, and `Risk_Metric`.
  - Computes risk count and material risk count.
  - Feeds ring charts and method/metric AI prompt context.

Imports:

- `utils.unique_join`

Used by:

- `main.py`
- `ai.py` indirectly through prompt facts passed from views.

Future development:

- Put count-based, deterministic analytics here.
- Keep GPT calls out of this file.
- If a new chart needs a reusable summary table, add a function here.

### `risk_dashboard/ai.py`

Purpose:

- Owns OpenAI Responses API calls.
- Builds prompt context.
- Generates GPT narrative text.
- Parses chatbot JSON responses.

Important functions:

- `selected_model()`
  - Returns `OPENAI_MODEL` or `DEFAULT_MODEL`.

- `require_openai_api_key()`
  - Requires `OPENAI_API_KEY`.
  - Raises a clear setup error when missing.

- `records_json(data, limit=None)`
  - Serializes dataframe rows to JSON safely.

- `build_inventory_context(data, limit=80)`
  - Selects `AI_CONTEXT_COLUMNS`.
  - Serializes filtered risk records for GPT.

- `count_records(data, columns, limit=20)`
  - Builds compact count records for prompt grounding.

- `build_taxonomy_prompt_facts(data, group_summary)`
  - Creates computed facts for Taxonomy L1 GPT summary.

- `build_compare_prompt_facts(data)`
  - Creates computed facts for Method and Metric GPT analysis.

- `format_chat_history(messages, limit=8)`
  - Condenses recent chatbot messages for follow-up context.

- `request_openai_text(model, system_prompt, user_prompt, max_output_tokens)`
  - Calls `client.responses.create()`.
  - Is cached through Streamlit for repeated analysis requests.

- `get_taxonomy_ai_summary(data, group_summary, model)`
  - Builds taxonomy prompt.
  - Calls GPT.
  - Returns Markdown.

- `get_compare_ai_analysis(data, model)`
  - Builds method/metric prompt.
  - Calls GPT.
  - Returns Markdown.

- `split_theme_ai_sections(text)`
  - Requires `## Theme Summary` and `## Theme Analysis`.
  - Splits GPT output into two card bodies.

- `get_theme_ai_analysis(data, themes, relationships, model)`
  - Builds theme prompt from theme mapping and relationship evidence.
  - Calls GPT.
  - Returns `(theme_summary, theme_analysis)`.

- `parse_json_object(text)`
  - Parses strict JSON or a JSON object embedded in a model response.

- `ask_inventory_chatbot(...)`
  - Sends filtered inventory plus chat history to GPT.
  - Requires JSON response with:
    - `answer_markdown`
    - `matching_group_ids`
  - Filters returned IDs to valid current-scope `Group_ID`s.

Imports:

- `settings.AI_CONTEXT_COLUMNS`
- `settings.CHAT_ASSOCIATED_RISK_LIMIT`
- `settings.DEFAULT_MODEL`
- `utils.format_top_counts`
- `utils.top_counts`

Used by:

- `main.py`
- `pages/2_AI_Risk_Theme_Classification.py`
- `views.py`

Future development:

- Add or revise GPT prompts here.
- Keep prompt facts grounded in computed data and filtered records.
- Do not expose model name, API key status, or source path in the UI.
- Do not add local replacement narrative logic here.
- If adding a new AI feature, create one public function that views can call.

### `risk_dashboard/themes.py`

Purpose:

- Owns AI risk theme classification.
- Converts risk records into semantic payloads.
- Generates OpenAI embeddings.
- Stores/searches vectors through FAISS.
- Computes hybrid similarity scores.
- Clusters risks into suggested themes.
- Builds theme records and review fields.

Main concepts:

- Official taxonomy remains authoritative.
- Suggested themes are an overlay only.
- Similarity is hybrid, not taxonomy-only:
  - title/description: 50%
  - metric/impact-comment: 25%
  - taxonomy: 15%
  - driver/root-cause: 10%

Important constants:

- `THEME_REVIEW_ACTIONS`
  - `Pending Review`, `Accept`, `Rename`, `Merge`, `Split`, `Reject`

- `DEFAULT_EMBEDDING_MODEL`
  - `text-embedding-3-large`

- `MIN_THEME_CONFIDENCE`
  - `0.50`

- `HYBRID_WEIGHTS`
  - score weights used by `hybrid_similarity_components()`

- `SIMILARITY_BANDS`
  - maps scores to:
    - near duplicate
    - same theme
    - related
    - weak relationship

Important function groups:

Field extraction:

- `field_text(row, column)`
- `root_cause_level(row)`
- `root_cause_driver(row)`
- `root_cause_evidence(row)`
- `clean_risk_title(row)`
- `risk_identifier(row)`

Payload construction:

- `build_standard_risk_text(row)`
- `build_semantic_payload(row)`

Token and topic helpers:

- `tokenize(text)`
- `semantic_keywords(text, limit=6)`
- `title_description_tokens(row)`
- `title_description_sections(row)`
- `common_topic(theme_data, limit=4)`
- `topic_adds_signal(topic, base_text)`
- `taxonomy_l2_is_broad(value)`
- `topic_should_split_theme(...)`

Embedding and vector store:

- `normalize_embedding_matrix(matrix)`
- `build_langchain_documents(data, payloads)`
- `build_openai_faiss_embeddings(data, payloads)`
- `build_embedding_layer(data, payloads)`

Similarity scoring:

- `calibrated_similarity(cosine_similarity)`
- `relation_for_score(score)`
- `token_similarity(left, right)`
- `taxonomy_alignment_score(source, target)`
- `driver_similarity_score(source, target)`
- `metric_impact_similarity_score(source, target)`
- `hybrid_similarity_components(source, target, semantic_similarity)`
- `build_hybrid_similarity_matrix(data, embeddings)`
- `build_similarity_relationships(...)`

Clustering:

- `find_parent(parent, item)`
- `union(parent, left, right)`
- `cluster_risks(...)`
- `initialize_centroids(...)`
- `cluster_risks_to_target(...)`
- `initialize_medoids(...)`
- `cluster_risks_to_target_by_similarity(...)`
- `merge_duplicate_theme_clusters(data, clusters)`

Theme records:

- `format_theme_label(value)`
- `dominant_value(series)`
- `unique_values(series, limit=6)`
- `title_fragment(value, max_words=4)`
- `theme_gap_text(theme_data)`
- `materiality_counts(theme_data)`
- `theme_key_drivers(theme_data, shared_topic)`
- `theme_review_flag(theme_data, confidence_score)`
- `average_internal_similarity(...)`
- `average_internal_similarity_from_matrix(...)`
- `confidence_profile(...)`
- `build_theme_records(...)`
- `uniquify_theme_names(themes)`

Page-level outputs:

- `build_dimension_count_summary(data)`
- `build_theme_classification(...)`

Imports:

- `langchain_core.documents.Document`
- `langchain_community.vectorstores.FAISS`
- `langchain_openai.OpenAIEmbeddings`
- `utils.unique_join`

Used by:

- `views.render_theme_classification()`

Future development:

- Add theme algorithm improvements here.
- Keep Streamlit rendering out of this file.
- Keep GPT narrative out of this file.
- If adding a new vector store later, add an explicit new function and call it from `build_embedding_layer()`.
- If adding persistent reviewer decisions, store/retrieve them outside this file, then pass the results into view or record-building functions.

### `risk_dashboard/views.py`

Purpose:

- Owns all Streamlit rendering.
- Converts dataframes and AI outputs into visible dashboard components.
- Handles user interaction state for review workflow and chatbot.

Important functions:

Shared rendering:

- `render_ai_card(title, markdown_body)`
  - Renders an AI narrative block.

- `render_ai_error(title, error)`
  - Renders an explicit AI error without replacing the result.

- `table_column_class(column)`
  - Converts column names into CSS class names.

- `format_compact_number(value)`
  - Formats numeric display values for tables.

- `render_table_cell(value, column, bar_columns, bool_columns)`
  - Adds score-bar and boolean-badge rendering.

- `render_html_table(data, max_rows=None, wide=False, bar_columns=None, bool_columns=None)`
  - Renders styled HTML tables.

Main dashboard:

- `metric_cards(data)`
  - Top-level count metrics.

- `render_group_summary(data, group_summary, model)`
  - Taxonomy L1 bar chart.
  - Taxonomy summary table.
  - AI taxonomy summary card.
  - Risk popover cards.

- `render_comparison(comparison, group_summary, data, model)`
  - Taxonomy L1 multiselect.
  - Assessment method ring chart.
  - Risk metric ring chart.
  - AI method/metric analysis card.

Theme page:

- `format_list_cell(values)`
- `wrap_chart_label(value, width=22)`
- `sync_theme_review_state(themes)`
- `render_theme_review_editor(themes)`
- `render_theme_review_controls(themes)`
- `render_theme_detail_cards(data, themes)`
- `render_count_ring_chart(summary, dimension, title, chart_height=220)`
- `render_theme_classification(data, model, show_heading=True)`

Chatbot:

- `render_risk_table(data, columns=None, max_rows=None)`
- `render_ai_chatbot(data, model)`

Imports:

- `ai.ask_inventory_chatbot`
- `ai.get_compare_ai_analysis`
- `ai.get_taxonomy_ai_summary`
- `ai.get_theme_ai_analysis`
- `settings.CHAT_ASSOCIATED_RISK_LIMIT`
- `settings.RISK_TABLE_COLUMNS`
- `themes.THEME_REVIEW_ACTIONS`
- `themes.build_theme_classification`
- `utils.unique_join`

Used by:

- `main.py`
- `pages/2_AI_Risk_Theme_Classification.py`

Future development:

- Add visual components here.
- Keep expensive computation out of views when it can live in `analysis.py` or `themes.py`.
- Keep OpenAI calls in `ai.py`; views should call public AI helper functions only.
- If adding new user controls, make sure state keys are stable and unique.

### `risk_dashboard/utils.py`

Purpose:

- Small formatting helpers used by multiple modules.

Important functions:

- `unique_join(series, limit=4)`
  - Joins unique non-empty values into compact labels.

- `top_counts(series, limit=2)`
  - Returns top value-count tuples.

- `format_top_counts(counts)`
  - Formats top-count tuples for prompt facts.

Used by:

- `analysis.py`
- `ai.py`
- `themes.py`
- `views.py`

Future development:

- Add tiny pure helper functions here only when used by multiple modules.
- Avoid adding Streamlit, OpenAI, or app orchestration here.

## 6. Data Lifecycle

```text
data/inventory.JSON
  -> data.load_json_records()
  -> data.records_from_payload()
  -> data.normalize_data()
     -> required field validation
     -> expected column ordering
     -> blank text cleanup
     -> generated Group_ID values
     -> Is_Material flag
     -> Likelihood_Score
  -> data.filter_data()
     -> shared sidebar filters
  -> filtered dataframe
     -> analysis.py summary tables
     -> views.py charts/tables
     -> ai.py GPT prompts
     -> themes.py embeddings/clustering
```

The filtered dataframe is the shared source of truth for each page render. All tabs and components should operate on the current filtered scope.

## 7. AI And Embedding Contract

This intern version uses a direct AI path.

Required:

- `OPENAI_API_KEY`
- `openai`
- `langchain-openai`
- `langchain-community`
- `faiss-cpu`

Optional environment overrides:

- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Rules:

- GPT summaries use the OpenAI Responses API.
- Theme embeddings use OpenAI embeddings and FAISS.
- If an AI call fails, the UI shows an explicit error for that output.
- Do not add local narrative replacement behavior unless the product owner asks for it.
- Do not show backend model/API status in the UI.
- Keep prompts grounded in computed facts and current filtered records.

## 8. Theme Classification Pipeline

```text
Filtered inventory dataframe
  -> semantic payload per risk
  -> OpenAI embeddings
  -> FAISS vector store
  -> cosine similarity matrix
  -> calibrated semantic score
  -> hybrid similarity score
  -> similar-risk relationship table
  -> target-count clustering
  -> duplicate-theme merge
  -> theme records
  -> dimension count summary
  -> GPT theme narrative
  -> Streamlit charts, detail cards, review workflow
```

Theme records include:

- `Theme_ID`
- `Theme_Name`
- `Theme_Summary`
- `Common_Topic`
- `Taxonomy_Alignment`
- `Business_Divisions`
- `GCRS_Values`
- `Risk_Metrics`
- `Assessment_Methods`
- `Risk_Count`
- `Material_Risks`
- `Non_Material_Risks`
- `Risk_IDs`
- `Confidence_Score`
- `Confidence_Rationale`
- `Governance_Check`
- `Potential_Gap`
- human review fields

Suggested themes are excluded from final display when:

- confidence is below `0.50`
- risk count is less than 2
- average internal similarity is zero

## 9. Streamlit State

State is used in two places:

1. Human review workflow:
   - `theme_review_state`
   - `review_name_{Theme_ID}`
   - `review_action_{Theme_ID}`
   - `reviewer_{Theme_ID}`
   - `review_notes_{Theme_ID}`

2. Chatbot:
   - `risk_chat_messages`
   - `risk_chat_selected_ids`

Current behavior:

- Review edits persist only for the current browser session.
- Chat history persists only for the current browser session.
- Associated-risk table follows the latest returned `Group_ID` scope.

Future persistence should be added as a new storage layer, not by hiding persistence code inside `views.py`.

## 10. Common Development Tasks

### Add A New Inventory Column

1. Add the column to `settings.EXPECTED_COLUMNS`.
2. If required, add it to `settings.REQUIRED_COLUMNS`.
3. If GPT should see it, add it to `settings.AI_CONTEXT_COLUMNS`.
4. If the chatbot table should show it, add it to `settings.RISK_TABLE_COLUMNS`.
5. Add any normalization rule in `data.normalize_data()`.
6. Add rendering in `views.py` only if visible UI is needed.

### Add A New Sidebar Filter

If every page should use the filter:

1. Add the column label mapping in `data.filter_data()`.
2. Confirm the column is in `settings.EXPECTED_COLUMNS`.

If only one page should use it:

1. Add the control inside the relevant `views.py` rendering function.
2. Apply it to the local dataframe scope before charts/tables.

### Add A New Main Dashboard Tab

1. Create a render function in `views.py`.
2. Create any summary builder in `analysis.py`.
3. Import the render function in `main.py`.
4. Add a new `st.tabs()` target in `main.py`.
5. Keep AI prompt functions in `ai.py` if the tab needs GPT.

### Add A New Top-Level Page

1. Create a file under `pages/`.
2. Keep that page file thin.
3. Use `configure_page()`, `inject_css()`, and `load_filtered_inventory()`.
4. Add the page to `app.py` navigation.
5. Put reusable rendering in `views.py`.

### Add A New GPT Analysis

1. Add one public function in `ai.py`.
2. Build computed facts before raw records.
3. Include `build_inventory_context()` only with needed columns and limits.
4. Preserve the instruction not to analyze `Impact_Numbers`.
5. Call the new AI function from `views.py`.
6. Handle exceptions with `render_ai_error()`.

### Add A New Theme Classification Stage

1. Add the pure algorithm function in `themes.py`.
2. Call it from `build_theme_classification()`.
3. Add output fields to `build_theme_records()` if needed.
4. Add display columns or charts in `views.py`.
5. Add prompt context in `ai.get_theme_ai_analysis()` if GPT should discuss the new field.

### Add Persistent Human Review

Recommended future shape:

```text
risk_dashboard/review_store.py
  -> load_review_decisions()
  -> save_review_decision()
  -> merge_review_decisions(themes)
```

Then:

1. `views.py` reads current reviewer inputs.
2. `review_store.py` persists decisions.
3. `themes.py` remains focused on generating suggested theme records.

Do not mix persistence code into theme clustering functions.

### Add Exports

Current export:

- Main page sidebar downloads `group_summary` CSV.

Recommended future export locations:

- Summary export button: `main.py` or `views.py`.
- Theme mapping export: `views.render_theme_classification()`.
- Export formatting helpers: new `exports.py` if exports become complex.

## 11. Future Stage Roadmap

### Stage 1: Current Intern Build

- Bundled JSON inventory.
- Shared sidebar filters.
- Count-based taxonomy and method/metric views.
- Required OpenAI GPT summaries.
- Required OpenAI/FAISS theme embeddings.
- Session-only human review.
- Session-only chatbot context.

### Stage 2: Reviewer Persistence

Goal:

- Save reviewer action, rename, reviewer name, notes, and date.

Likely files:

- New `risk_dashboard/review_store.py`
- Updates to `views.render_theme_review_controls()`
- Optional updates to `settings.py` for storage path constants.

Do not change:

- Theme clustering logic in `themes.py`.
- Official taxonomy fields.

### Stage 3: Managed Vector Store

Goal:

- Replace local FAISS with managed vector storage or metadata filtering.

Likely files:

- `themes.py`
- Possible new `vector_store.py`
- `requirements.txt`
- `README.md`

Design idea:

```text
themes.build_embedding_layer()
  -> vector_store.build_embeddings()
  -> vector_store.search_similar_risks()
```

Keep the output contract the same:

- embedding matrix or relationship records
- valid `Group_ID` traceability
- no official taxonomy overwrite

### Stage 4: Governance Workflow

Goal:

- Approval status, audit trail, reviewer assignment, exportable evidence.

Likely files:

- New `review_store.py`
- New `exports.py`
- Updates to `views.py`
- Possibly updates to `settings.py`

Keep:

- `Theme_ID`
- `Risk_IDs`
- `Governance_Check`
- `Confidence_Rationale`

### Stage 5: Larger Inventory Performance

Goal:

- Handle larger risk inventories without slow page reloads.

Likely changes:

- Cache embedding calls by stable payload hash.
- Move relationship generation into a service function.
- Add precomputed artifacts.
- Add tests around clustering thresholds.

Likely files:

- `themes.py`
- possible new `cache.py`
- possible new `tests/`

### Stage 6: Multi-Source Data

Goal:

- Load inventory from approved internal sources, not only bundled JSON.

Likely files:

- New `data_sources.py`
- Updates to `bootstrap.py`
- Updates to `settings.py`

Important:

- Do not expose data source controls in the front end unless product requirements change.
- Keep normalized dataframe schema stable.

## 12. Guardrails

Keep these rules intact:

- `app.py` and page files stay thin.
- `data.py` owns loading, schema normalization, and global filters.
- `analysis.py` owns deterministic summary tables.
- `ai.py` owns GPT prompts and OpenAI calls.
- `themes.py` owns classification algorithms.
- `views.py` owns Streamlit rendering and UI state.
- `styles.py` owns CSS and page config.
- `settings.py` owns constants and shared column lists.
- Official taxonomy fields are source-of-truth fields.
- AI theme fields are suggested overlays.
- Do not analyze `Impact_Numbers`.
- Do not expose model/API/source details in the UI.
- Do not add hidden local AI substitutes unless explicitly requested.

## 13. Verification

Run from the project folder:

```bash
cd "/Users/ke/Documents/New project 3/risk_taxonomy_dashboard_Intern"
```

Syntax check:

```bash
python -m compileall app.py pages risk_dashboard
```

Data and deterministic summary check:

```bash
python - <<'PY'
from risk_dashboard.data import load_json_records, normalize_data
from risk_dashboard.analysis import build_group_summary, build_method_metric

df = normalize_data(load_json_records())
print(len(df), "records")
print(len(build_group_summary(df)), "Taxonomy L1 groups")
print(len(build_method_metric(df)), "method/metric rows")
PY
```

Expected result for the bundled inventory at the time this guide was written:

```text
20 records
7 Taxonomy L1 groups
19 method/metric rows
```

Full app smoke test:

```bash
export OPENAI_API_KEY="your-api-key"
streamlit run app.py
```

Manual checks:

- Main dashboard loads.
- Sidebar filters affect every tab.
- Taxonomy L1 chart and table update after filters.
- Method and metric rings update after filters.
- AI summaries show GPT output or explicit AI errors.
- AI chatbot returns an answer and filters the associated-risk table.
- AI Risk Theme Classification generates themes when OpenAI embeddings are available.
- Human review controls update session state.
- Similar Risk Relationships table shows traceable risk pairs.

## 14. Troubleshooting

Missing API key:

```text
OPENAI_API_KEY is required for AI analysis in the intern dashboard.
```

Fix:

```bash
export OPENAI_API_KEY="your-api-key"
```

Missing dependency:

```text
ModuleNotFoundError: No module named ...
```

Fix:

```bash
pip install -r requirements.txt
```

Missing required columns:

The app stops in `data.normalize_data()` and reports the missing column names.

Fix:

- Add the missing fields to `data/inventory.JSON`, or
- Update `settings.REQUIRED_COLUMNS` only if product requirements changed.

No records after filters:

The app stops in `bootstrap.load_filtered_inventory()`.

Fix:

- Clear or broaden sidebar filters.

Theme page is slow:

Likely cause:

- Embedding calls and pairwise similarity generation.

Future optimization options:

- cache embedding results
- precompute embeddings
- reduce filtered scope
- add a managed vector store

## 15. Mental Model For New Work

When adding anything, first ask:

1. Is it data loading or schema cleanup?
   - Use `data.py` and `settings.py`.

2. Is it deterministic analytics?
   - Use `analysis.py`.

3. Is it GPT text, chatbot behavior, or prompt construction?
   - Use `ai.py`.

4. Is it risk theme similarity, clustering, embeddings, or theme records?
   - Use `themes.py`.

5. Is it a chart, table, card, control, expander, or Streamlit state?
   - Use `views.py`.

6. Is it visual styling?
   - Use `styles.py` or `.streamlit/config.toml`.

7. Is it page routing?
   - Use `app.py` or a file in `pages/`.

If a change seems to belong in many places, start by defining the data contract in `settings.py`, the transformation in one domain module, and the rendering in `views.py`. That keeps the project teachable and easier to extend.
