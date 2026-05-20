# Risk Invex — Modern Enterprise Streamlit Template

A Bloomberg Terminal-lite Streamlit template for risk inventory, QoQ monitoring, data quality checks, and AI risk analysis.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Included

- Dark enterprise dashboard layout
- Sidebar page navigation
- KPI cards
- Interactive risk slicer row
- ECharts heatmap
- ECharts bar chart
- Context-aware AI Risk Analyst chat
- Emerging risk alerts
- Latest risk change table
- Custom CSS styling

## Recommended Next Modules

```text
services/
  risk_service.py       # load and search risk inventory
  qoq_service.py        # compare quarter-over-quarter changes
  dq_service.py         # data quality checks
  ai_service.py         # AI summary and risk explanation
  export_service.py     # PDF / Excel export
```

## Design Direction

This is optimized for a Risk Invex / AI Risk Inventory MVP:
- dense but readable
- dark executive theme
- minimal emoji
- strong card layout
- ECharts for professional visuals
- Python-first backend logic
