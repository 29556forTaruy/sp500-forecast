#!/usr/bin/env python
"""Phase 6/7 — daily forecast engine.

Ties the validated ladder together (Level-0 drift + Level-5 GARCH/FHS distribution,
PLAN.md §11/§13) and emits the day's forecast as JSON for the app / daily batch.
This is the production output schema (PLAN §7.3): latest fan distribution, the
indicator panel (value + z-score + percentile, for the heatmap), and a backfilled
prediction-vs-realized history (the answer-key log, PLAN §0 — populated from the
walk-forward backtest so the app has content from day one).

Outputs (app/):
  forecast.json   asof, spot, 12m quantiles, monthly fan path, indicators, provenance
  history.json    past forecasts (annual) with realized outcome + 90%-band hit flag
Run: uv run python forecast.py   (the GitHub Actions daily job runs exactly this)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase5_fanchart import (load_inputs, predictive_quantiles, latest_forecast,
                             fan_from_fhs, calib_metrics, QGRID)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
APP = ROOT / "app"

# indicator panel: (master column, label, "high value =" direction for equities)
INDICATORS = [
    ("shiller_CAPE", "CAPE (Shiller P/E)", "bearish"),
    ("T10Y2Y", "Yield curve 10y-2y", "bullish"),    # inversion (low/neg) = bearish
    ("DGS10", "10y Treasury yield", "neutral"),
    ("FEDFUNDS", "Fed funds rate", "bearish"),
    ("VIXCLS", "VIX", "bearish"),
    ("NFCI", "Financial conditions (NFCI)", "bearish"),  # higher = tighter = bearish
    ("BAMLH0A0HYM2", "High-yield credit spread", "bearish"),
    ("UNRATE", "Unemployment rate", "neutral"),
    ("INDPRO", "Industrial production (YoY)", "bullish"),
]


def indicator_panel(master: pd.DataFrame) -> list:
    out = []
    for col, label, direction in INDICATORS:
        s = master.set_index("date")[col].dropna()
        if s.empty:
            continue
        val = float(s.iloc[-1])
        if col == "INDPRO":  # show as YoY %
            s = (np.log(s) - np.log(s.shift(12))).dropna() * 100
            val = float(s.iloc[-1])
        trail = s.iloc[-120:] if len(s) >= 24 else s
        z = float((val - trail.mean()) / trail.std()) if trail.std() > 0 else 0.0
        pct = float((s < val).mean())
        out.append({"key": col, "label": label, "value": round(val, 2),
                    "z_10y": round(z, 2), "pctile": round(pct, 3),
                    "direction": direction, "asof": str(s.index[-1].date())})
    return out


def build_history(d, realized, mu, quant, best_idx_model, P0_series, n_per_year=1):
    """Backfilled answer-key: for ~one origin per year with a known 12m outcome,
    record the forecast median / 90% band (price) and whether the realized price
    landed inside the 90% interval."""
    dates = d.index
    rows = []
    seen_years = set()
    qlo, qmd, qhi = 0, np.where(np.isclose(QGRID, 0.5))[0][0], len(QGRID) - 1  # 0.05,0.5,0.95
    q25 = int(np.where(np.isclose(QGRID, 0.25))[0][0])  # inner 50% band for the time-machine fan
    q75 = int(np.where(np.isclose(QGRID, 0.75))[0][0])
    for i in range(len(d)):
        y = dates[i].year
        if y in seen_years or np.isnan(quant[best_idx_model][i, 0]):
            continue
        seen_years.add(y)
        p0 = float(P0_series.iloc[i])
        med = p0 * np.exp(quant[best_idx_model][i, qmd])
        lo = p0 * np.exp(quant[best_idx_model][i, qlo])
        hi = p0 * np.exp(quant[best_idx_model][i, qhi])
        lo50 = p0 * np.exp(quant[best_idx_model][i, q25])
        hi50 = p0 * np.exp(quant[best_idx_model][i, q75])
        realized_px = p0 * np.exp(realized[i])
        rows.append({"origin": str(dates[i].date()), "spot": round(p0, 1),
                     "fc_median": round(med, 1), "fc_lo90": round(lo, 1), "fc_hi90": round(hi, 1),
                     "fc_lo50": round(lo50, 1), "fc_hi50": round(hi50, 1),
                     "realized": round(realized_px, 1),
                     "realized_ret_pct": round((np.exp(realized[i]) - 1) * 100, 1),
                     "in90": bool(lo <= realized_px <= hi)})
    return rows


CASE_STUDIES = [
    ("1999-12-31", "Dot-com peak (Dec 1999)"),
    ("2007-10-31", "Pre-GFC peak (Oct 2007)"),
    ("2009-03-31", "GFC trough (Mar 2009)"),
    ("2020-02-29", "Pre-COVID (Feb 2020)"),
]


def case_studies(d, realized, quant, model, p0_series):
    """Forecasts the model made at famous turning points + what actually happened —
    the honest record including the misses (e.g. it could not foresee 2008)."""
    dates = d.index
    qmd = int(np.where(np.isclose(QGRID, 0.5))[0][0])
    rows = []
    for tgt, label in CASE_STUDIES:
        i = int(np.argmin(np.abs(dates - pd.Timestamp(tgt))))
        if np.isnan(quant[model][i, 0]):
            continue
        p0 = float(p0_series.iloc[i])
        rows.append({"label": label, "origin": str(dates[i].date()), "spot": round(p0, 1),
                     "fc_median": round(p0 * np.exp(quant[model][i, qmd]), 1),
                     "fc_lo90": round(p0 * np.exp(quant[model][i, 0]), 1),
                     "fc_hi90": round(p0 * np.exp(quant[model][i, -1]), 1),
                     "realized": round(p0 * np.exp(realized[i]), 1),
                     "realized_ret_pct": round((np.exp(realized[i]) - 1) * 100, 1),
                     "in90": bool(quant[model][i, 0] <= realized[i] <= quant[model][i, -1])})
    return rows


def main():
    APP.mkdir(exist_ok=True)
    print("running validated ladder (Level 0 drift + Level 5 vol models) ...")
    df, bt, disp = load_inputs()
    d, realized, mu, quant, sigma_cal, pit, models = predictive_quantiles(df, bt, disp)

    # pick best vol model by pinball on the long common window
    valid = {m: ~np.isnan(quant[m][:, 0]) for m in models}
    common = np.logical_and.reduce([valid[m] for m in ["const", "ewma", "garch"]])
    pin = {m: calib_metrics(realized, quant[m], pit[m], QGRID, common)["pinball"]
           for m in ["const", "ewma", "garch"]}
    best = min(pin, key=pin.get)
    cov = calib_metrics(realized, quant[best], pit[best], QGRID, common)

    # latest live forecast + fan
    mu_L, sig_L, zpool, P0, asof = latest_forecast(df, d, realized, mu, sigma_cal, best)
    months, bands, q12 = fan_from_fhs(mu_L, sig_L, zpool, P0)

    master = pd.read_csv(OUT / "master_monthly.csv", parse_dates=["date"])
    p0_series = pd.Series(np.exp(df.set_index("date")["log_p"].reindex(d.index).to_numpy()), index=d.index)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]

    pit_best = pit[best][common]
    pit_best = pit_best[~np.isnan(pit_best)]
    pit_counts, _ = np.histogram(pit_best, bins=10, range=(0, 1))
    forecast = {
        "asof": asof,
        "spot": round(P0, 1),
        "horizon_months": 12,
        "model": {"drift": "Level-0 structural anchor (log PERxEPS)",
                  "vol": best, "shape": "filtered historical simulation",
                  "mu_12m_log": round(mu_L, 4), "sigma_12m": round(sig_L, 4)},
        "return_quantiles_pct": {str(q): round((np.exp(mu_L + sig_L * np.quantile(zpool, q)) - 1) * 100, 1) for q in qs},
        "price_quantiles": {str(q): round(v, 1) for q, v in q12.items()},
        "fan_path": {"months": months.tolist(),
                     **{f"q{int(q*100):02d}": [round(x, 1) for x in bands[q]] for q in qs}},
        "indicators": indicator_panel(master),
        "calibration": {"window": [str(d.index[common].min().date()), str(d.index[common].max().date())],
                        "n": int(common.sum()),
                        "cover50": round(cov["cover_50"], 3), "cover80": round(cov["cover_80"], 3),
                        "cover90": round(cov["cover_90"], 3),
                        "pit_ks": round(cov["pit_ks"], 3),
                        "pit_hist": pit_counts.tolist(),
                        "note": "walk-forward calibrated; outer bands reliable"},
        "case_studies": case_studies(d, realized, quant, best, p0_series),
    }
    (APP / "forecast.json").write_text(json.dumps(forecast, indent=2))

    history = build_history(d, realized, mu, quant, best, p0_series)
    hit = np.mean([h["in90"] for h in history]) if history else float("nan")
    (APP / "history.json").write_text(json.dumps(
        {"note": "Backfilled from walk-forward backtest (answer-key log). 'in90' = realized inside the 90% band.",
         "n": len(history), "hit_rate_90": round(float(hit), 3), "records": history}, indent=2))

    print(f"  asof {asof}: spot {P0:,.0f} -> 12m median {q12[0.5]:,.0f} "
          f"({(q12[0.5]/P0-1)*100:+.1f}%), 90% [{q12[0.05]:,.0f}, {q12[0.95]:,.0f}]")
    print(f"  vol={best}, calibration cover90={cov['cover_90']:.2f}, history records={len(history)} "
          f"(90% hit-rate {hit:.0%})")
    print(f"  -> app/forecast.json, app/history.json")


if __name__ == "__main__":
    main()
