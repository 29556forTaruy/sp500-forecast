#!/usr/bin/env python
"""Benchmark leaderboard — how does this model really compare to the world's standard
equity-forecast models, walk-forward and honestly?

POINT forecasts vs the expanding HISTORICAL MEAN (the Goyal-Welch 2008 reference):
  random walk(=0), historical mean, drift-only, our Level-0 anchor, and univariate
  predictive regressions on the classic predictors — dividend yield (D/P), CAPE yield
  (1/CAPE), earnings yield (E/P), term spread (10y-2y) — plus a Goyal-Welch kitchen-sink.
  Scored by OOS R² vs the mean, Clark-West nested test, and direction hit-rate.

DISTRIBUTION: our GARCH+FHS fan vs a Gaussian / Student-t / unconditional-historical fan
  on the SAME drift and σ (only the shape differs), scored by pinball / coverage / PIT.

Honest by construction: the `verdict` is computed from the data (cw_p, oos_r2), never
hardcoded — at 1y almost nothing beats the mean; at long horizons CAPE-yield wins.
Output: app/benchmarks.json (read by the app's ⑨ Benchmark tab). Runs on its own cadence
(weekly), not in the daily job. Run: uv run python benchmarks.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from phase3_level0 import build_features, backtest, MIN_TRAIN
from phase4_level2 import clark_west, direction_hit
from phase5_fanchart import vol_raw_const, vol_raw_ewma, vol_raw_garch, calib_metrics, QGRID
from forecast import HORIZONS, horizon_walk_forward

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
APP = ROOT / "app"

LB_HORIZONS = ["12mo", "36mo", "60mo", "120mo"]   # shows the 1y→10y ranking flip
POINT_LABELS = {
    "rw_zero": "Random walk (no drift)", "hist_mean": "Historical mean",
    "drift_only": "EPS-growth drift", "level0": "Level-0 (this model)",
    "div_yield": "Dividend yield (D/P)", "cape_yield": "CAPE yield (1/CAPE)",
    "earn_yield": "Earnings yield (E/P)", "term_spread": "Term spread (10y-2y)",
    "goyal_welch": "Goyal-Welch kitchen sink",
}


def build_predictors(df, master):
    """Classic predictors aligned to the feature (Shiller month-end) dates, lag 0 to match
    the model's headline backtest information set."""
    m = master.set_index("date")
    idx = df["date"]
    dp = (m["shiller_D"] / m["shiller_P"]).reindex(idx).to_numpy()
    ep = (m["shiller_E"] / m["shiller_P"]).reindex(idx).to_numpy()
    cy = (1.0 / df["CAPE"]).to_numpy()
    ts = m["T10Y2Y"].reindex(idx).to_numpy()
    return {"div_yield": dp, "earn_yield": ep, "cape_yield": cy, "term_spread": ts}


def _wf_regression(fwd, preds, origin_pos, H, min_train):
    """Walk-forward predictive regression: at each origin i, OLS of completed H-month
    returns on the lagged predictor(s) over data ≤ i-H, then predict at i. preds is an
    (n, k) matrix (k=1 univariate, k>1 kitchen sink)."""
    fc = np.full(len(origin_pos), np.nan)
    P = preds if preds.ndim == 2 else preds[:, None]
    for s, i in enumerate(origin_pos):
        jmax = i - H
        if jmax < 0:
            continue
        j = np.arange(0, jmax + 1)
        ok = ~np.isnan(fwd[j]) & ~np.isnan(P[j]).any(axis=1)
        j = j[ok]
        if len(j) < min_train or np.isnan(P[i]).any():
            continue
        X = np.column_stack([np.ones(len(j)), P[j]])
        beta, *_ = np.linalg.lstsq(X, fwd[j], rcond=None)
        fc[s] = float(beta[0] + P[i] @ beta[1:])
    return fc


def point_leaderboard(df, predictors, H):
    bt = backtest(df, lag=0, H=H).reset_index(drop=True)
    dates = bt["date"]
    realized = bt["realized"].to_numpy()
    hist_mean = bt["fc_rw_drift"].to_numpy()          # the Goyal-Welch reference benchmark
    # map bt origins back to positions in df (predictors are indexed like df)
    pos = pd.Index(df["date"]).get_indexer(dates)
    fwd_full = (df["log_p"].shift(-H) - df["log_p"]).to_numpy()

    fcs = {"rw_zero": bt["fc_rw_zero"].to_numpy(), "hist_mean": hist_mean,
           "drift_only": bt["fc_drift_only"].to_numpy(), "level0": bt["fc_level0_est"].to_numpy()}
    for name, p in predictors.items():
        fcs[name] = _wf_regression(fwd_full, p, pos, H, MIN_TRAIN)
    ks = np.column_stack([predictors[k] for k in ["div_yield", "earn_yield", "cape_yield", "term_spread"]])
    fcs["goyal_welch"] = _wf_regression(fwd_full, ks, pos, H, MIN_TRAIN)

    rows = []
    for name, fc in fcs.items():
        mask = ~np.isnan(fc) & ~np.isnan(realized) & ~np.isnan(hist_mean)
        if mask.sum() < 24:
            continue
        r, f, hm = realized[mask], fc[mask], hist_mean[mask]
        sse_m = float(np.sum((r - f) ** 2)); sse_b = float(np.sum((r - hm) ** 2))
        oos_r2 = 1 - sse_m / sse_b if sse_b > 0 else float("nan")
        # the project's honest window: full-sample R² is inflated by the data-poor pre-1950
        # era (PLAN §11), so also report post-1950 where most predictability evaporates.
        m50 = mask & (dates.dt.year.to_numpy() >= 1950)
        r2_50 = None
        if m50.sum() >= 24:
            sb50 = float(np.sum((realized[m50] - hist_mean[m50]) ** 2))
            r2_50 = round(1 - float(np.sum((realized[m50] - fc[m50]) ** 2)) / sb50, 3) if sb50 > 0 else None
        if name == "hist_mean":
            cw_t = cw_p = None; verdict = "reference"
        else:
            cw = clark_west(r, hm, f)
            cw_t, cw_p = round(cw["cw_t"], 2), round(cw["cw_p_onesided"], 3)
            verdict = ("beats_mean" if (cw_p < 0.05 and oos_r2 > 0)
                       else "tie" if oos_r2 > 0 else "loses_to_mean")
        wdates = dates[mask]
        rows.append({"model": name, "oos_r2": round(oos_r2, 3), "oos_r2_post1950": r2_50,
                     "cw_t": cw_t, "cw_p": cw_p,
                     "dir_hit": round(direction_hit(r, f), 3), "verdict": verdict,
                     "n": int(mask.sum()), "n_eff": max(1, int(mask.sum() / H)),
                     "window": [str(wdates.min().date()), str(wdates.max().date())]})
    rows.sort(key=lambda x: (-(x["oos_r2"] if x["oos_r2"] == x["oos_r2"] else -9), x["model"]))
    return rows


def dist_leaderboard(df, raw_s, spec):
    H = spec["H"]; min_cal = spec["min_cal"]
    d, mu, realized, quant, sigma_cal, pit = horizon_walk_forward(df, raw_s, spec)
    sg = sigma_cal["garch"]; n = len(mu)
    common = ~np.isnan(quant["garch"][:, 0])
    znorm = stats.norm.ppf(QGRID)
    nu = 5.0; sc = np.sqrt((nu - 2) / nu); zt = stats.t.ppf(QGRID, nu) * sc
    qg, pg = quant["garch"], pit["garch"]
    qga = np.full_like(qg, np.nan); pga = np.full(n, np.nan)
    qt = np.full_like(qg, np.nan); pt = np.full(n, np.nan)
    qh = np.full_like(qg, np.nan); ph = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(sg[i]):
            std = (realized[i] - mu[i]) / sg[i]
            qga[i] = mu[i] + sg[i] * znorm; pga[i] = float(stats.norm.cdf(std))
            qt[i] = mu[i] + sg[i] * zt;     pt[i] = float(stats.t.cdf(std / sc, nu))
        done = np.arange(0, max(i - H + 1, 0))
        rd = realized[done] - mu[done]; rd = rd[~np.isnan(rd)]
        if len(rd) >= min_cal:
            qh[i] = mu[i] + np.quantile(rd, QGRID)
            ph[i] = float(np.mean(rd <= (realized[i] - mu[i])))
    rows = []
    for name, q, p in [("garch_fhs", qg, pg), ("gaussian", qga, pga),
                       ("student_t", qt, pt), ("historical", qh, ph)]:
        mask = common & np.isfinite(q).all(axis=1)
        if mask.sum() < 24:
            continue
        c = calib_metrics(realized, q, p, QGRID, mask)
        rows.append({"model": name, "pinball": round(c["pinball"], 4),
                     "cover90": round(c["cover_90"], 3), "pit_ks": round(c["pit_ks"], 3), "n": c["n"]})
    rows.sort(key=lambda r: r["pinball"])
    if rows:
        rows[0]["verdict"] = "best"
    return rows


def main():
    print("benchmark leaderboard: this model vs the world's standard equity forecasts ...")
    df = build_features().sort_values("date").reset_index(drop=True)
    df["r_m"] = df["log_p"].diff()
    master = pd.read_csv(OUT / "master_monthly.csv", parse_dates=["date"])
    predictors = build_predictors(df, master)
    rm = df["r_m"].to_numpy()
    raw_s = {"const": pd.Series(vol_raw_const(rm), index=df["date"]),
             "ewma": pd.Series(vol_raw_ewma(rm), index=df["date"]),
             "garch": pd.Series(vol_raw_garch(rm), index=df["date"])}

    horizons = {}
    for spec in [s for s in HORIZONS if s["key"] in LB_HORIZONS]:
        key = spec["key"]; H = spec["H"]
        point = point_leaderboard(df, predictors, H)
        dist = dist_leaderboard(df, raw_s, spec)
        by = {r["model"]: r for r in point}
        lv, dr = by.get("level0"), by.get("drift_only")
        val_incr = round(lv["oos_r2"] - dr["oos_r2"], 3) if lv and dr else None
        val_incr_50 = (round(lv["oos_r2_post1950"] - dr["oos_r2_post1950"], 3)
                       if lv and dr and lv["oos_r2_post1950"] is not None and dr["oos_r2_post1950"] is not None else None)
        preds = ["div_yield", "cape_yield", "earn_yield", "term_spread", "goyal_welch"]
        beating = [m for m in preds if by.get(m, {}).get("verdict") == "beats_mean"]
        horizons[key] = {"horizon_months": H, "point": point, "distribution": dist,
                         "headline": {"valuation_increment": val_incr, "valuation_increment_post1950": val_incr_50,
                                      "n_predictors_beating_mean": len(beating), "n_predictors": len(preds),
                                      "n_eff": lv["n_eff"] if lv else None,
                                      "dist_best": dist[0]["model"] if dist else None}}
        print(f"  [{key:5s}] level0 R²{lv['oos_r2']:+.3f} (post1950 {lv['oos_r2_post1950']}) | "
              f"valuation increment {val_incr} (post1950 {val_incr_50}) | "
              f"{len(beating)}/{len(preds)} predictors beat mean | dist best {dist[0]['model'] if dist else '-'}")

    out = {"schema_version": 1, "asof": str(df["date"].iloc[-1].date()),
           "reference": "historical_mean",
           "indices": {"SP500": {"label": "S&P 500", "horizons": horizons}}}
    (APP / "benchmarks.json").write_text(json.dumps(out, indent=2))
    print(f"  -> app/benchmarks.json ({list(horizons.keys())})")


if __name__ == "__main__":
    main()
