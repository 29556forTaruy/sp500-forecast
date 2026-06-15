"""S&P 500 1-year probabilistic forecast — Streamlit app (PLAN.md §8).

Reads the daily-batch output (forecast.json, history.json) produced by
`uv run python forecast.py`. Tabs: ① fan chart, ② indicator heatmap,
③ calibration (is the model honest?), ④ answer-key log + crisis case studies.
No modeling here — pure presentation. Run: uv run streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
st.set_page_config(page_title="S&P 500 1-year forecast", page_icon="📈", layout="wide")

# plain-language notes per indicator (shown as hover help on the heatmap)
INDICATOR_HELP = {
    "shiller_CAPE": "Price ÷ 10-year inflation-adjusted earnings. High = expensive = lower long-run returns. The most reliable long-horizon valuation gauge — but weak at 1 year.",
    "T10Y2Y": "10-year minus 2-year Treasury yield. Negative (inverted curve) has preceded most recessions; steep/positive = expansion.",
    "DGS10": "10-year Treasury yield. Higher rates pressure valuations but can also signal growth — direction-neutral on its own.",
    "FEDFUNDS": "Fed policy rate. Rising = tightening (headwind); falling = easing (tailwind).",
    "VIXCLS": "Option-implied 30-day volatility, the 'fear gauge'. Spikes in stress; low = calm (current fan is narrow because VIX is moderate).",
    "NFCI": "Chicago Fed financial-conditions index. Positive = tighter than average financial conditions (headwind).",
    "BAMLH0A0HYM2": "High-yield bond spread over Treasuries. Widening = credit stress; tight = risk appetite.",
    "UNRATE": "Unemployment rate. A sharp rise (Sahm rule) flags recession.",
    "INDPRO": "Industrial production, year-over-year %. Negative = industrial contraction.",
}


@st.cache_data
def load():
    fc = json.loads((APP / "forecast.json").read_text())
    hist = json.loads((APP / "history.json").read_text())
    return fc, hist


fc, hist = load()
spot = fc["spot"]; q = fc["price_quantiles"]; rq = fc["return_quantiles_pct"]; cal = fc["calibration"]

st.title("S&P 500 — 1-year probabilistic forecast")
st.caption(f"As of {fc['asof']} · spot {spot:,.0f} · drift = {fc['model']['drift']} · "
           f"vol = {fc['model']['vol']} · shape = {fc['model']['shape']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot:,.0f}")
c2.metric("12m median", f"{q['0.5']:,.0f}", f"{rq['0.5']:+.1f}%")
c3.metric("90% range", f"{q['0.05']:,.0f} – {q['0.95']:,.0f}")
c4.metric("50% range", f"{q['0.25']:,.0f} – {q['0.75']:,.0f}")

st.markdown(
    f"**Plain English:** a year from now the model's *typical* outcome is **{rq['0.5']:+.1f}%** "
    f"(≈ {q['0.5']:,.0f}), with a 1-in-2 chance of landing between **{rq['0.25']:+.1f}%** and "
    f"**{rq['0.75']:+.1f}%**, and a 9-in-10 chance between **{rq['0.05']:+.1f}%** and **{rq['0.95']:+.1f}%**. "
    f"The median sits above the average because crashes drag the *mean* down — most years are modestly up, "
    f"a few are sharply down. Valuation (CAPE) is historically extreme, which lowers the *mean* but barely "
    f"moves the 1-year median; it mainly fattens the downside.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["① Fan chart", "② Indicator heatmap", "③ Calibration", "④ Answer-key log"])

# ----------------------------------------------------------------- fan chart
with tab1:
    fp = fc["fan_path"]; m = [0] + fp["months"]
    def path(qk): return [spot] + fp[qk]
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
    st.info(f"Walk-forward calibrated {cal['window'][0]}–{cal['window'][1]} (n={cal['n']}): the 80% band "
            f"historically covered {cal['cover80']:.0%} and the 90% band {cal['cover90']:.0%} of outcomes "
            f"(targets 80% / 90%). Drift = Level-0 structural anchor; width = GARCH conditional volatility; "
            f"shape = empirical (filtered historical simulation), so the downside tail is fatter than the upside.")

# ---------------------------------------------------------------- heatmap
with tab2:
    df = pd.DataFrame(fc["indicators"])
    def stance(r):
        s = r["z_10y"]
        return -s if r["direction"] == "bearish" else (s if r["direction"] == "bullish" else 0.0)
    df["stance"] = df.apply(stance, axis=1)
    df["help"] = df["key"].map(INDICATOR_HELP)
    df["percentile"] = (df["pctile"] * 100).round().astype(int)
    show = df[["label", "value", "z_10y", "percentile", "direction", "stance", "help"]]

    def stance_css(v):  # red↔green wash without needing matplotlib (Streamlit Cloud stays light)
        t = max(-1.0, min(1.0, v / 2.0))
        rgb = (40, 170, 70) if t >= 0 else (210, 60, 60)
        return f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},{abs(t)*0.55+0.08:.2f})"

    st.dataframe(
        show.style.map(stance_css, subset=["stance"])
        .format({"value": "{:.2f}", "z_10y": "{:+.2f}", "stance": "{:+.2f}", "percentile": "{:d}%"}),
        use_container_width=True, hide_index=True,
        column_config={
            "label": st.column_config.TextColumn("Indicator"),
            "z_10y": st.column_config.NumberColumn("z (10y)", help="Standard deviations from the trailing 10-year mean"),
            "percentile": st.column_config.TextColumn("%ile", help="Rank within full history"),
            "direction": st.column_config.TextColumn("high =", help="What a high reading means for equities"),
            "stance": st.column_config.NumberColumn("stance", help="Signed z: green supports equities, red is a headwind"),
            "help": st.column_config.TextColumn("what it is", width="large"),
        })
    st.caption("`stance` = signed z-score (green = currently supportive of equities, red = headwind). "
               "Hover any cell for detail. CAPE near its all-time-high percentile is the dominant red flag, "
               "but valuation is a *long-horizon* signal — it shapes the distribution more than the 1-year median.")

# ---------------------------------------------------------------- calibration
with tab3:
    st.subheader("Is the model honest?")
    st.write("A forecast distribution is **calibrated** if outcomes fall inside its bands as often as "
             "claimed. We check this walk-forward (only past data at each date) — the project's #1 rule.")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("50% band covered", f"{cal.get('cover50', float('nan')):.0%}", "target 50%")
    cc2.metric("80% band covered", f"{cal['cover80']:.0%}", "target 80%")
    cc3.metric("90% band covered", f"{cal['cover90']:.0%}", "target 90%")
    pit = cal.get("pit_hist", [])
    if pit:
        n = sum(pit); k = len(pit); ideal = n / k
        figp = go.Figure()
        figp.add_trace(go.Bar(x=[f"{i*10}–{(i+1)*10}%" for i in range(k)], y=pit, marker_color="#1f77b4",
                              name="observed"))
        figp.add_hline(y=ideal, line_dash="dash", line_color="grey", annotation_text="uniform (ideal)")
        figp.update_layout(height=360, xaxis_title="PIT bucket (where the outcome fell in the forecast)",
                           yaxis_title="count", margin=dict(t=20), showlegend=False)
        st.plotly_chart(figp, use_container_width=True)
        st.caption(f"**PIT histogram** — if the distribution is perfectly calibrated this is flat (each bucket "
                   f"≈ {ideal:.0f}). Bars near the edges that are too tall would mean over-confidence (outcomes "
                   f"hit the tails too often); too short means the bands are too wide. Ours is close to flat "
                   f"(KS={cal['pit_ks']}), mildly imperfect — honest about its own limits.")

# ---------------------------------------------------------------- answer-key
with tab4:
    st.subheader("Track record (answer-key log)")
    h = pd.DataFrame(hist["records"])
    st.write(f"{hist['n']} past forecasts; the realized index landed inside the 90% band "
             f"**{hist['hit_rate_90']:.0%}** of the time (target 90%). Kept visible by design (PLAN §0).")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_hi90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_lo90"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                              line=dict(width=0), name="90% band (forecast)"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_median"], line=dict(color="#1f77b4", dash="dash"), name="forecast median"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["realized"], mode="markers",
                              marker=dict(color=["#2ca02c" if v else "#d62728" for v in h["in90"]], size=7),
                              name="realized (green = in band)"))
    fig2.update_layout(height=400, xaxis_title="forecast origin", yaxis_title="S&P 500 level (1y later)",
                       margin=dict(t=20), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Crisis case studies — including the misses")
    st.write("How the model called famous turning points, and what actually happened 12 months later:")
    cs = pd.DataFrame(fc.get("case_studies", []))
    if not cs.empty:
        cols = st.columns(len(cs))
        for col, (_, r) in zip(cols, cs.iterrows()):
            verdict = "✅ in 90% band" if r["in90"] else "❌ outside band (a real miss)"
            col.markdown(f"**{r['label']}**")
            col.metric(f"actual 1y", f"{r['realized_ret_pct']:+.0f}%",
                       f"forecast median {((r['fc_median']/r['spot'])-1)*100:+.0f}%")
            col.caption(f"made: {r['fc_lo90']:,.0f}–{r['fc_hi90']:,.0f} → actual {r['realized']:,.0f}  \n{verdict}")
        st.caption("The October-2007 row is the honest headline: the model forecast a normal year and the "
                   "index fell ~37% into the GFC — **no valuation/vol model foresaw 2008**. The wide bands and "
                   "the answer-key exist precisely so the model never pretends otherwise.")
    st.dataframe(h[["origin", "spot", "fc_median", "fc_lo90", "fc_hi90", "realized", "realized_ret_pct", "in90"]]
                 .rename(columns={"realized_ret_pct": "realized %"}).iloc[::-1],
                 use_container_width=True, hide_index=True)
