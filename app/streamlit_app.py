"""S&P 500 1-year probabilistic forecast — Streamlit app (PLAN.md §8).

Reads the daily-batch output (forecast.json, history.json) produced by
`uv run python forecast.py`. Three screens: ① fan chart, ② indicator heatmap,
③ prediction-vs-realized answer-key log. No modeling here — pure presentation.
Run: uv run streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
st.set_page_config(page_title="S&P 500 1-year forecast", layout="wide")


@st.cache_data
def load():
    fc = json.loads((APP / "forecast.json").read_text())
    hist = json.loads((APP / "history.json").read_text())
    return fc, hist


fc, hist = load()
spot = fc["spot"]; q = fc["price_quantiles"]; rq = fc["return_quantiles_pct"]

st.title("S&P 500 — 1-year probabilistic forecast")
st.caption(f"As of {fc['asof']} · spot {spot:,.0f} · drift = {fc['model']['drift']} · "
           f"vol = {fc['model']['vol']} · shape = {fc['model']['shape']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:,.0f}")
c2.metric("12m median", f"{q['0.5']:,.0f}", f"{rq['0.5']:+.1f}%")
c3.metric("90% range", f"{q['0.05']:,.0f} – {q['0.95']:,.0f}")
c4.metric("50% range", f"{q['0.25']:,.0f} – {q['0.75']:,.0f}")

tab1, tab2, tab3 = st.tabs(["① Fan chart", "② Indicator heatmap", "③ Answer-key log"])

# ----------------------------------------------------------------- fan chart
with tab1:
    fp = fc["fan_path"]
    m = [0] + fp["months"]
    def path(qk):
        return [spot] + fp[qk]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=path("q95"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m, y=path("q05"), fill="tonexty", fillcolor="rgba(31,119,180,0.13)",
                             line=dict(width=0), name="5–95%"))
    fig.add_trace(go.Scatter(x=m, y=path("q75"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=m, y=path("q25"), fill="tonexty", fillcolor="rgba(31,119,180,0.28)",
                             line=dict(width=0), name="25–75%"))
    fig.add_trace(go.Scatter(x=m, y=path("q50"), line=dict(color="#1f77b4", width=3), name="median"))
    fig.add_hline(y=spot, line_dash="dot", line_color="grey", annotation_text=f"now {spot:,.0f}")
    fig.update_layout(height=480, xaxis_title="months ahead", yaxis_title="S&P 500 level",
                      margin=dict(t=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    cal = fc["calibration"]
    st.info(f"Walk-forward calibrated {cal['window'][0]}–{cal['window'][1]} (n={cal['n']}): "
            f"80% band covers {cal['cover80']:.0%}, 90% band covers {cal['cover90']:.0%} "
            f"(targets 80% / 90%). The median ({rq['0.5']:+.1f}%) sits above the mean drift because "
            f"1-year equity returns are left-skewed — typical years beat the crash-dragged mean.")

# ---------------------------------------------------------------- heatmap
with tab2:
    df = pd.DataFrame(fc["indicators"])
    # signed stance: how bullish(+)/bearish(-) the current reading is for equities
    def stance(r):
        s = r["z_10y"]
        if r["direction"] == "bearish":
            return -s
        if r["direction"] == "bullish":
            return s
        return 0.0
    df["stance"] = df.apply(stance, axis=1)
    df["percentile"] = (df["pctile"] * 100).round().astype(int).astype(str) + "%"
    show = df[["label", "value", "z_10y", "percentile", "direction", "stance"]].rename(
        columns={"label": "indicator", "z_10y": "z (10y)", "direction": "high = "})
    st.dataframe(
        show.style.background_gradient(cmap="RdYlGn", subset=["stance"], vmin=-2, vmax=2)
        .format({"value": "{:.2f}", "z (10y)": "{:+.2f}", "stance": "{:+.2f}"}),
        use_container_width=True, hide_index=True)
    st.caption("`stance` = signed z-score (green = currently supportive of equities, red = headwind). "
               "CAPE near its historical extreme is the dominant red flag; valuation is a long-horizon "
               "signal, so it shapes the distribution more than the 1-year median.")

# ---------------------------------------------------------------- answer-key
with tab3:
    h = pd.DataFrame(hist["records"])
    st.write(f"**Backfilled answer-key** — {hist['n']} past forecasts; "
             f"realized landed inside the 90% band **{hist['hit_rate_90']:.0%}** of the time (target 90%).")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_hi90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_lo90"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                              line=dict(width=0), name="90% band (forecast)"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_median"], line=dict(color="#1f77b4", dash="dash"), name="forecast median"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["realized"], mode="markers",
                              marker=dict(color=["#2ca02c" if v else "#d62728" for v in h["in90"]], size=7),
                              name="realized (green=in band)"))
    fig2.update_layout(height=420, xaxis_title="forecast origin", yaxis_title="S&P 500 level (1y later)",
                       margin=dict(t=20), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(h[["origin", "spot", "fc_median", "fc_lo90", "fc_hi90", "realized", "realized_ret_pct", "in90"]]
                 .rename(columns={"realized_ret_pct": "realized %"}).iloc[::-1],
                 use_container_width=True, hide_index=True)
    st.caption("Each row: the forecast made at `origin` and what the index actually did 12 months later. "
               "This is the model's honest track record — kept visible by design (PLAN §0).")
