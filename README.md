# RISKEX Streamlit Demo

A Streamlit prototype inspired by the Inv_chat architecture.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Optional AI

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.2"
streamlit run app.py
```

Without an API key, the chat panel runs in demo mode.

## Files

- `app.py` — main Streamlit UI
- `sample_risk_inventory.json` — sample local risk inventory
- `requirements.txt` — Python dependencies
