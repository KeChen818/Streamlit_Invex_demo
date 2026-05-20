import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts

st.set_page_config(
    page_title="Risk Invex | Enterprise Dashboard",
    page_icon="▣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load CSS
# -----------------------------
def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# -----------------------------
# Mock data
# -----------------------------
risk_data = pd.DataFrame([
    {"Risk": "Third-party cyber breach", "Category": "Technology", "Impact": "High", "Likelihood": "High", "Change": "New", "Owner": "Technology Risk"},
    {"Risk": "AI model bias compliance", "Category": "Compliance", "Impact": "High", "Likelihood": "Medium", "Change": "Increased", "Owner": "Compliance"},
    {"Risk": "Vendor concentration", "Category": "Operational", "Impact": "Medium", "Likelihood": "Medium", "Change": "New", "Owner": "Ops Risk"},
    {"Risk": "Cloud cost escalation", "Category": "Financial", "Impact": "Medium", "Likelihood": "High", "Change": "Increased", "Owner": "Finance"},
    {"Risk": "Data quality degradation", "Category": "Operational", "Impact": "Medium", "Likelihood": "Medium", "Change": "Decreased", "Owner": "Data Governance"},
])

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("<div class='brand'>RISK INVEX</div>", unsafe_allow_html=True)
    st.markdown("<div class='go-button'>&lt; GO &gt;</div>", unsafe_allow_html=True)

    st.markdown("### Dashboard")
    st.page_link("app.py", label="Executive Summary")
    st.markdown("Risk Heatmap")
    st.markdown("QoQ Changes")

    st.markdown("### Inventory")
    st.markdown("Risk Library")
    st.markdown("Controls")
    st.markdown("Issues")

    st.markdown("### Intelligence")
    st.markdown("AI Risk Analyst")
    st.markdown("Scenario Analysis")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="top-header">
  <div>
    <div class="eyebrow">Enterprise Risk Command Center</div>
    <div class="title">Risk Inventory Intelligence</div>
  </div>
  <div class="header-pill">Q2 2026 Active Cycle</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Ticker row
# -----------------------------
st.markdown("""
<div class="ticker-row">
  <div class="ticker-card"><span>SPX</span><b class="down">5,278.40 -0.47%</b></div>
  <div class="ticker-card"><span>VIX</span><b class="up">12.45 +1.22%</b></div>
  <div class="ticker-card"><span>USD</span><b class="up">105.32 +0.18%</b></div>
  <div class="ticker-card"><span>10Y</span><b class="down">4.28 -0.03%</b></div>
  <div class="ticker-card"><span>Cycle</span><b class="up">Q2'26 Open</b></div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# KPI cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
kpis = [
    ("Total Risks", "248", "▲ 14 vs last quarter", "up"),
    ("High Risks", "38", "▲ 6 vs last quarter", "down"),
    ("QoQ Changes", "37", "11 new / 5 resolved", "neutral"),
    ("DQ Score", "92%", "▲ 3pp vs last quarter", "up"),
]

for col, (label, value, foot, cls) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-foot {cls}">{foot}</div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Charts
# -----------------------------
left, right = st.columns([1.35, 0.85])

with left:
    chart_a, chart_b = st.columns(2)

    with chart_a:
        st.markdown("<div class='section-title'>Risk Heatmap</div>", unsafe_allow_html=True)
        heatmap_option = {
            "tooltip": {},
            "grid": {"top": 35, "right": 20, "bottom": 40, "left": 65},
            "xAxis": {
                "type": "category",
                "data": ["Very Low", "Low", "Medium", "High", "Very High"],
                "axisLabel": {"color": "#9ca3af"},
            },
            "yAxis": {
                "type": "category",
                "data": ["Very Low", "Low", "Medium", "High", "Very High"],
                "axisLabel": {"color": "#9ca3af"},
            },
            "visualMap": {
                "min": 0,
                "max": 18,
                "orient": "horizontal",
                "left": "center",
                "bottom": 0,
                "textStyle": {"color": "#9ca3af"},
                "inRange": {"color": ["#14532d", "#eab308", "#dc2626"]},
            },
            "series": [{
                "type": "heatmap",
                "data": [[0,0,0],[1,0,1],[2,0,4],[3,0,7],[4,0,9],
                         [0,1,1],[1,1,3],[2,1,7],[3,1,11],[4,1,14],
                         [0,2,2],[1,2,5],[2,2,12],[3,2,18],[4,2,10],
                         [0,3,1],[1,3,2],[2,3,6],[3,3,7],[4,3,3],
                         [0,4,0],[1,4,0],[2,4,1],[3,4,1],[4,4,1]],
                "label": {"show": True, "color": "#ffffff"},
            }],
        }
        st_echarts(heatmap_option, height="340px")

    with chart_b:
        st.markdown("<div class='section-title'>Top Risk Categories</div>", unsafe_allow_html=True)
        bar_option = {
            "tooltip": {},
            "grid": {"top": 20, "right": 25, "bottom": 25, "left": 95},
            "xAxis": {"type": "value", "axisLabel": {"color": "#9ca3af"}},
            "yAxis": {
                "type": "category",
                "inverse": True,
                "data": ["Operational", "Technology", "Compliance", "Strategic", "Financial"],
                "axisLabel": {"color": "#9ca3af"},
            },
            "series": [{
                "type": "bar",
                "data": [65, 48, 38, 27, 24],
                "barWidth": 16,
                "itemStyle": {"borderRadius": [0, 8, 8, 0], "color": "#f59e0b"},
                "label": {"show": True, "position": "right", "color": "#f5f5f5"},
            }]
        }
        st_echarts(bar_option, height="340px")

    st.markdown("<div class='section-title'>Latest Risk Changes</div>", unsafe_allow_html=True)
    st.dataframe(risk_data, use_container_width=True, hide_index=True)

with right:
    st.markdown("""
    <div class="insight-card">
      <div class="section-title">AI Risk Analyst</div>
      <p>
        Elevated third-party technology risk detected, driven by vendor concentration,
        weak access-control evidence, and external incident signals.
      </p>
      <div class="tag amber">Suggested Review</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
      <div class="section-title">Emerging Risks</div>
      <div class="alert-row"><span>AI model hallucination risk</span><b class="risk-high">High</b></div>
      <div class="alert-row"><span>Shadow AI usage</span><b class="risk-high">High</b></div>
      <div class="alert-row"><span>Critical vendor instability</span><b class="risk-med">Medium</b></div>
      <div class="alert-row"><span>Cloud cost escalation</span><b class="risk-med">Medium</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
      <div class="section-title">Recent Alerts</div>
      <div class="alert-row"><span>Unusual data movement pattern</span><em>10m</em></div>
      <div class="alert-row"><span>Failed access review</span><em>35m</em></div>
      <div class="alert-row"><span>DQ rule breach</span><em>1h</em></div>
    </div>
    """, unsafe_allow_html=True)