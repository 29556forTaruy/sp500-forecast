#!/usr/bin/env python
"""Phase 4 — Level 2: regularized linear model (ElasticNet) in RESIDUAL mode.

PLAN.md §5 ladder, Level 2. The Level-0 anchor (Phase 3) showed that valuation
alone adds ~zero at the 1-year horizon. This asks the next question: do the FRED
MACRO indicators (yield curve, credit/financial conditions, labor, activity, money)
add any 1-year predictive signal ON TOP of the Level-0 anchor?

Design (every choice made to defeat look-ahead and overfitting — the two killers):
  * Target = residual = realized 12m log return − Level-0 anchor forecast (residual
    mode, so ElasticNet shrinks toward the structural anchor, not toward zero).
  * Features = point-in-time macro transforms, each lagged by its realistic
    publication delay (market/rates 1m, macro releases 2m, valuation 3m).
  * Walk-forward: refit every 12 months on the EXPANDING window of completed
    (features, residual) pairs; StandardScaler fit on the TRAIN fold only;
    ElasticNetCV picks alpha/l1_ratio by CV inside the train fold only.
  * Evaluated OOS on the SAME origins against: Level-0 anchor, drift_only,
    rw_drift — Level 2 ships only if it beats Level 0 OOS (plan philosophy).

Outputs:
  data/processed/level2_features_monthly.csv   the lagged macro feature matrix
  data/processed/level2_backtest_summary.json  OOS metrics, coefficients
  data/processed/level2_backtest.png
Run: uv run python phase4_level2.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV

from phase3_level0 import build_features, backtest as level0_backtest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"

H = 12
REFIT_EVERY = 12     # months between model re-estimations (slow layer; plan: monthly/quarterly)
MIN_TRAIN = 180      # min completed (feature, residual) pairs before first fit (~15y)
L1_GRID = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

# (master column, transform, publication lag in months). Transforms are point-in-time.
FEATURE_SPEC = [
    ("val_gap",      "as_is", 3),   # from level0 features (valuation reversion gap)
    ("cape_pctile",  "as_is", 3),   # CAPE percentile in trailing 30y
    ("T10Y2Y",       "as_is", 1),   # yield-curve slope (10y-2y)
    ("DGS10",        "chg12", 1),   # 10y yield 12m change
    ("FEDFUNDS",     "chg12", 2),   # policy rate 12m change
    ("NFCI",         "as_is", 1),   # Chicago Fed financial conditions (already an index)
    ("UNRATE",       "chg12", 2),   # unemployment 12m change (Sahm-style)
    ("INDPRO",       "yoy",   2),   # industrial production YoY
    ("M2SL",         "yoy",   2),   # money supply YoY
]
MODERN_EXTRA = [("VIXCLS", "as_is", 1)]  # adds VIX but only from 1990


def transform(s: pd.Series, kind: str) -> pd.Series:
    if kind == "as_is":
        return s
    if kind == "chg12":
        return s - s.shift(12)
    if kind == "yoy":
        return np.log(s) - np.log(s.shift(12))
    raise ValueError(kind)


def build_feature_matrix(spec) -> pd.DataFrame:
    m = pd.read_csv(OUT / "master_monthly.csv", parse_dates=["date"]).set_index("date")
    lf = pd.read_csv(OUT / "level0_features_monthly.csv", parse_dates=["date"]).set_index("date")
    m["val_gap"] = lf["val_gap"]
    m["cape_pctile"] = lf["cape_pctile_360m"]
    feat = pd.DataFrame(index=m.index)
    for col, kind, lag in spec:
        feat[f"{col}_{kind}"] = transform(m[col], kind).shift(lag)  # lag = publication delay
    return feat


def clark_west(realized, fc_restricted, fc_unrestricted, nwlag=18) -> dict:
    """Nested-model OOS test (Clark-West 2007): is the unrestricted model (Level 2)
    a genuine improvement over the nested restricted model (Level 0)? Corrects the
    upward bias in the MSPE difference. f_t regressed on a constant with NW SE."""
    e0 = realized - fc_restricted
    e1 = realized - fc_unrestricted
    f = e0 ** 2 - (e1 ** 2 - (fc_restricted - fc_unrestricted) ** 2)
    Xc = np.ones((len(f), 1))
    res = sm.OLS(f, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": nwlag})
    return {"cw_mean": float(f.mean()), "cw_t": float(res.tvalues[0]),
            "cw_p_onesided": float(res.pvalues[0] / 2)}


def direction_hit(realized, fc) -> float:
    return float(np.mean(np.sign(fc) == np.sign(realized)))


def run(spec, tag: str, residual_mode: bool = True) -> dict:
    # Level-0 per-origin forecasts (residual target base / benchmark)
    df = build_features()
    bt = level0_backtest(df, lag=0).set_index("date")
    feat = build_feature_matrix(spec)

    data = bt.join(feat, how="inner").dropna(subset=[c for c in feat.columns])
    data = data.dropna(subset=["realized", "fc_level0_est"])
    fcols = list(feat.columns)
    # residual mode: learn realized − anchor (shrink toward anchor). standalone:
    # learn the absolute return directly (does macro predict returns at all?).
    resid = ((data["realized"] - data["fc_level0_est"]) if residual_mode
             else data["realized"]).to_numpy()
    X = data[fcols].to_numpy()
    dates = data.index.to_numpy()
    n = len(data)

    preds = np.full(n, np.nan)        # predicted residual at each origin
    coef_log = []
    model = None
    scaler_mean = scaler_std = None
    for i in range(n):
        # train on completed pairs: origin j with j's 12m target done by origin i (j <= i-H)
        train_idx = np.array([j for j in range(i) if j <= i - H])
        if len(train_idx) < MIN_TRAIN:
            continue
        if model is None or i % REFIT_EVERY == 0:
            Xtr, ytr = X[train_idx], resid[train_idx]
            scaler_mean, scaler_std = Xtr.mean(0), Xtr.std(0)
            scaler_std[scaler_std == 0] = 1.0
            Xtr_s = (Xtr - scaler_mean) / scaler_std
            model = ElasticNetCV(l1_ratio=L1_GRID, cv=5, max_iter=5000, n_jobs=-1)
            model.fit(Xtr_s, ytr)
            coef_log.append({"date": str(pd.Timestamp(dates[i]).date()),
                             "alpha": float(model.alpha_), "l1": float(model.l1_ratio_),
                             "n_train": int(len(train_idx)),
                             **{c: float(v) for c, v in zip(fcols, model.coef_)}})
        x_s = (X[i] - scaler_mean) / scaler_std
        preds[i] = float(model.predict(x_s.reshape(1, -1))[0])

    ok = ~np.isnan(preds)
    d = data.iloc[ok].copy()
    p = preds[ok]
    realized = d["realized"].to_numpy()
    # residual mode: forecast = anchor + predicted residual; standalone: forecast = prediction
    level2_fc = (d["fc_level0_est"].to_numpy() + p) if residual_mode else p

    def sse(fc):
        return float(np.sum((realized - fc) ** 2))

    benches = {
        "level0": d["fc_level0_est"].to_numpy(),
        "drift_only": d["fc_drift_only"].to_numpy(),
        "rw_drift": d["fc_rw_drift"].to_numpy(),
    }
    s2 = sse(level2_fc)
    metrics = {
        "tag": tag, "mode": "residual" if residual_mode else "standalone",
        "n_origins": int(ok.sum()),
        "date_range": [str(d.index.min().date()), str(d.index.max().date())],
        "rmse_level2": float(np.sqrt(np.mean((realized - level2_fc) ** 2))),
        "rmse_level0": float(np.sqrt(np.mean((realized - benches["level0"]) ** 2))),
        "oos_r2_level2_vs_level0": 1 - s2 / sse(benches["level0"]),
        "oos_r2_level2_vs_drift": 1 - s2 / sse(benches["drift_only"]),
        "oos_r2_level2_vs_rw_drift": 1 - s2 / sse(benches["rw_drift"]),
        # is the Level-2 'improvement' over the nested Level-0 distinguishable from 0?
        "clark_west_vs_level0": clark_west(realized, benches["level0"], level2_fc),
        "dir_hit_level2": direction_hit(realized, level2_fc),
        "dir_hit_level0": direction_hit(realized, benches["level0"]),
        "n_refits": len(coef_log),
        "mean_abs_coef": {c: float(np.mean([abs(cl[c]) for cl in coef_log])) for c in fcols},
        "frac_nonzero": {c: float(np.mean([cl[c] != 0 for cl in coef_log])) for c in fcols},
    }
    # post-1990 cut for fairness vs the modern variant
    yr = d.index.year.to_numpy()
    if (yr >= 1990).sum() > 50:
        m90 = yr >= 1990
        s2_90 = float(np.sum((realized[m90] - level2_fc[m90]) ** 2))
        b90 = float(np.sum((realized[m90] - benches["level0"][m90]) ** 2))
        metrics["oos_r2_level2_vs_level0_post1990"] = 1 - s2_90 / b90
    return {"metrics": metrics, "coef_log": coef_log,
            "series": {"date": [str(x) for x in d.index.date],
                       "realized": realized.tolist(),
                       "level0": benches["level0"].tolist(),
                       "level2": level2_fc.tolist()}}


def dispersion_diag(spec) -> dict:
    """Carry-forward check (audit): macro may predict DISPERSION/risk, not the mean.
    Correlate each point-in-time feature with the ABSOLUTE Level-0 forecast error —
    a crude realized-volatility proxy. This is the Level-5 (fan-chart width) target."""
    df = build_features()
    bt = level0_backtest(df, lag=0).set_index("date")
    feat = build_feature_matrix(spec)
    d = bt.join(feat, how="inner").dropna(subset=list(feat.columns) + ["realized", "fc_level0_est"])
    abs_err = (d["realized"] - d["fc_level0_est"]).abs()
    return {c: float(d[c].corr(abs_err)) for c in feat.columns}


def main():
    # NOTE: 'core' is labeled 1976 by feature availability, but the binding feature
    # T10Y2Y (1976-07) + MIN_TRAIN make the actual OOS evaluation window 1992–2025.
    print("[1/4] core model, residual mode (9 features) ...")
    core = run(FEATURE_SPEC, "core", residual_mode=True)
    print("[2/4] core model, STANDALONE (no anchor) — fairness check ...")
    standalone = run(FEATURE_SPEC, "core_standalone", residual_mode=False)
    print("[3/4] modern model (+VIX, from 2006 eval) ...")
    modern = run(FEATURE_SPEC + MODERN_EXTRA, "modern", residual_mode=True)
    disp = dispersion_diag(FEATURE_SPEC)

    feat = build_feature_matrix(FEATURE_SPEC)
    feat.reset_index().to_csv(OUT / "level2_features_monthly.csv", index=False)
    summary = {"spec": {"H": H, "refit_every": REFIT_EVERY, "min_train": MIN_TRAIN,
                        "eval_window_note": "core OOS window is 1992-2025 (T10Y2Y starts 1976-07 + MIN_TRAIN)"},
               "core_residual": core["metrics"], "core_standalone": standalone["metrics"],
               "modern_residual": modern["metrics"],
               "macro_vs_absError_corr_for_level5": disp}
    (OUT / "level2_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    print("  -> level2_features_monthly.csv, level2_backtest_summary.json")

    # chart
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    s = core["series"]
    dts = pd.to_datetime(s["date"])
    ax[0].plot(dts, s["realized"], "k.", ms=3, label="realized")
    ax[0].plot(dts, s["level0"], "C0-", lw=1, label="Level 0 anchor")
    ax[0].plot(dts, s["level2"], "C3-", lw=1, label="Level 2 (anchor+macro)")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_title("Level 2 vs Level 0 — 12m log return forecast (core, 1976+)")
    ax[0].legend(fontsize=8)

    cm = core["metrics"]
    bars = {"vs Level0": cm["oos_r2_level2_vs_level0"],
            "vs drift_only": cm["oos_r2_level2_vs_drift"],
            "vs RW+drift": cm["oos_r2_level2_vs_rw_drift"]}
    ax[1].bar(list(bars), list(bars.values()),
              color=["C0" if v < 0 else "C2" for v in bars.values()])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Level 2 OOS R² (>0 = adds value over that baseline)")
    for i, v in enumerate(bars.values()):
        ax[1].text(i, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout(); fig.savefig(OUT / "level2_backtest.png", dpi=110)
    print("  -> level2_backtest.png")

    print("[4/4] digest")
    for name, r in [("core residual", core["metrics"]), ("core STANDALONE", standalone["metrics"]),
                    ("modern+VIX", modern["metrics"])]:
        cw = r["clark_west_vs_level0"]
        print(f"\n  {name}: n={r['n_origins']} ({r['date_range'][0]}..{r['date_range'][1]})")
        print(f"    OOS R² vs Level0={r['oos_r2_level2_vs_level0']:+.4f}  "
              f"vs drift={r['oos_r2_level2_vs_drift']:+.4f}  vs RW+drift={r['oos_r2_level2_vs_rw_drift']:+.4f}")
        print(f"    Clark-West vs Level0: t={cw['cw_t']:.2f} (p={cw['cw_p_onesided']:.2f}) "
              f"→ {'detectable' if cw['cw_p_onesided'] < 0.05 else 'NOT distinguishable from 0'}")
        print(f"    direction hit: Level2={r['dir_hit_level2']:.1%} vs Level0={r['dir_hit_level0']:.1%}")
    print("\n  macro vs |Level-0 error| corr (Level-5 risk carry-forward):")
    for k, v in sorted(disp.items(), key=lambda kv: -abs(kv[1]))[:4]:
        print(f"    {k:22s} {v:+.3f}")


if __name__ == "__main__":
    main()
