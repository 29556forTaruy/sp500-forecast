#!/usr/bin/env python
"""Phase 3 — Level 0 structural anchor: feature build + walk-forward backtest.

Implements PLAN.md §5.1. The model (in log space) is
    ln P_hat(t+12) = ln P_t + g_t + lambda * (ln CAPE*_t - ln CAPE_t)
i.e. EPS-growth drift plus partial reversion of the multiple toward its
trailing median. We backtest several lambda policies — including the user's
original full-reversion (lambda=1) idea — against random-walk baselines,
strictly walk-forward (every quantity at origin t uses only data <= t).

Outputs:
  data/processed/level0_features_monthly.csv   point-in-time features + fwd label
  data/processed/level0_backtest_summary.json  metrics for each policy/variant
  data/processed/level0_backtest.png           backtest + lambda-sensitivity chart
Run: uv run python phase3_level0.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"

H = 12          # forecast horizon (months)
MED_WIN = 360   # CAPE* / quantile window (30y), trailing
G_WIN = 240     # EPS-growth window (20y), trailing
MIN_TRAIN = 120 # min completed (val_gap, fwd_ret) pairs before estimating lambda
LAG = 3         # earnings publication lag for the honest variant (months)
LAMBDA_PRIOR = 0.10
LAMBDA_CAP = 0.35
QS = [0.05, 0.25, 0.75, 0.95]


# ----------------------------------------------------------------- features

def build_features() -> pd.DataFrame:
    sh = pd.read_csv(OUT / "shiller_monthly.csv", parse_dates=["date"])
    df = sh.loc[sh["CAPE"].notna(), ["date", "P", "CAPE"]].reset_index(drop=True)

    df["log_p"] = np.log(df["P"])
    df["log_cape"] = np.log(df["CAPE"])
    df["e10n"] = df["P"] / df["CAPE"]                 # smoothed EPS, time-t dollars (§5.1)
    df["log_e10n"] = np.log(df["e10n"])

    # 20y annualized log growth of smoothed earnings (trailing only)
    df["g20_e10n"] = (df["log_e10n"] - df["log_e10n"].shift(G_WIN)) / (G_WIN / 12)

    # trailing 30y median & quantiles of CAPE (closed-left: only past data)
    roll = df["CAPE"].rolling(MED_WIN, min_periods=MED_WIN)
    df["cape_star_360m"] = roll.median()
    for q in QS:
        df[f"cape_q{int(q*100):02d}_360m"] = roll.quantile(q)
    df["cape_pctile_360m"] = df["CAPE"].rolling(MED_WIN, min_periods=MED_WIN).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)

    df["val_gap"] = np.log(df["cape_star_360m"]) - df["log_cape"]

    # the only forward-looking column (label); NaN for the last H rows
    df["fwd12_log_ret"] = df["log_p"].shift(-H) - df["log_p"]
    return df


# ------------------------------------------------------------ walk-forward

def estimate_lambda(y: np.ndarray, x: np.ndarray) -> float:
    """OLS through the origin of y on x: slope = lambda (reversion speed)."""
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else np.nan


def backtest(df: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
    """One forecast per monthly origin. `lag` shifts the EARNINGS-derived signal
    (val_gap, g) by `lag` months — the price anchor log_p stays current — to
    emulate the publication lag of Shiller earnings (§5.1 honest variant)."""
    g = df["g20_e10n"].shift(lag).to_numpy()
    vg = df["val_gap"].shift(lag).to_numpy()
    log_p = df["log_p"].to_numpy()
    fwd = df["fwd12_log_ret"].to_numpy()
    # quantile band offsets (in log price), using lagged signal too
    q_off = {q: (np.log(df[f"cape_q{int(q*100):02d}_360m"]) - np.log(df["cape_star_360m"])).shift(lag).to_numpy()
             for q in QS}

    n = len(df)
    rows = []
    # index of completed-target rows usable for training at each origin
    for i in range(n):
        if i + H >= n or np.isnan(fwd[i]) or np.isnan(vg[i]) or np.isnan(g[i]):
            continue
        # training set: rows j whose target completed by origin i (j + H <= i)
        train_mask = np.zeros(n, dtype=bool)
        jmax = i - H
        if jmax < 0:
            continue
        j = np.arange(0, jmax + 1)
        valid = ~np.isnan(fwd[j]) & ~np.isnan(vg[j]) & ~np.isnan(g[j])
        j = j[valid]
        if len(j) < MIN_TRAIN:
            continue
        y_tr = fwd[j] - g[j]
        x_tr = vg[j]
        lam_ols = estimate_lambda(y_tr, x_tr)
        lam_shrink = float(np.clip(0.5 * lam_ols + 0.5 * LAMBDA_PRIOR, 0.0, LAMBDA_CAP))
        rw_drift = float(np.mean(fwd[j]))  # expanding historical mean annual log return

        realized = fwd[i]
        fcs = {
            "level0_est": g[i] + lam_shrink * vg[i],
            "level0_prior": g[i] + LAMBDA_PRIOR * vg[i],
            "full_reversion": g[i] + 1.0 * vg[i],     # user's original idea (lambda=1)
            "drift_only": g[i],                        # EPS growth, no valuation
            "rw_zero": 0.0,
            "rw_drift": rw_drift,
        }
        rec = {"date": df["date"].iloc[i], "realized": realized,
               "lam_ols": lam_ols, "lam_used": lam_shrink, **{f"fc_{k}": v for k, v in fcs.items()}}
        # band (κ=1) around the level0_est point forecast
        pt = fcs["level0_est"]
        for q in QS:
            rec[f"band_q{int(q*100):02d}"] = pt + lam_shrink * q_off[q][i] if not np.isnan(q_off[q][i]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def metrics(bt: pd.DataFrame) -> dict:
    r = bt["realized"].to_numpy()
    policies = ["level0_est", "level0_prior", "full_reversion", "drift_only", "rw_zero", "rw_drift"]
    err = {p: r - bt[f"fc_{p}"].to_numpy() for p in policies}
    sse = {p: float(np.sum(err[p] ** 2)) for p in policies}
    out = {"n_origins": int(len(bt)),
           "date_range": [bt["date"].min().date().isoformat(), bt["date"].max().date().isoformat()]}
    for p in policies:
        e = err[p]
        out[p] = {
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "oos_r2_vs_rw_zero": 1 - sse[p] / sse["rw_zero"],
            "oos_r2_vs_rw_drift": 1 - sse[p] / sse["rw_drift"],
            "mean_signed_err": float(np.mean(e)),
        }
    # band coverage (q05..q95 should contain ~90% if calibrated)
    lo, hi = bt["band_q05"].to_numpy(), bt["band_q95"].to_numpy()
    inside = (r >= lo) & (r <= hi)
    out["band_cover_90"] = float(np.nanmean(inside.astype(float)))
    lo25, hi75 = bt["band_q25"].to_numpy(), bt["band_q75"].to_numpy()
    out["band_cover_50"] = float(np.nanmean(((r >= lo25) & (r <= hi75)).astype(float)))
    return out


def honest_metrics(bt: pd.DataFrame) -> dict:
    """The framing-critical numbers (per audit): the headline full-sample OOS R²
    is inflated by the weak pre-1950 expanding-mean benchmark, and almost all of
    the edge is the EPS-growth drift, not valuation. So we report (a) sub-period
    OOS R² vs rw_drift, (b) the valuation increment = level0 vs the drift_only
    baseline, and (c) the share of SSE improvement coming from pre-1950."""
    r = bt["realized"].to_numpy()
    yr = bt["date"].dt.year.to_numpy()

    def oos_r2(col, bench, mask=None):
        m = np.ones(len(bt), bool) if mask is None else mask
        e_m = (r[m] - bt[f"fc_{col}"].to_numpy()[m]) ** 2
        e_b = (r[m] - bt[f"fc_{bench}"].to_numpy()[m]) ** 2
        return float(1 - e_m.sum() / e_b.sum())

    out = {}
    for tag, mask in [("full", None), ("post1950", yr >= 1950), ("post1960", yr >= 1960)]:
        out[tag] = {
            "level0_est_vs_rw_drift": oos_r2("level0_est", "rw_drift", mask),
            "drift_only_vs_rw_drift": oos_r2("drift_only", "rw_drift", mask),
            # valuation increment: does the CAPE term beat pure EPS-growth drift?
            "level0_est_vs_drift_only": oos_r2("level0_est", "drift_only", mask),
        }
    # share of total SSE improvement (vs rw_drift) coming from pre-1950
    e_m = (r - bt["fc_level0_est"].to_numpy()) ** 2
    e_b = (r - bt["fc_rw_drift"].to_numpy()) ** 2
    imp = e_b - e_m
    out["pre1950_share_of_improvement"] = float(imp[yr < 1950].sum() / imp.sum())
    return out


def insample_lambda_nw(df: pd.DataFrame) -> dict:
    """Full-sample lambda with Newey-West(18) t-stat — in-sample diagnostic (§5.1/§4)."""
    d = df.dropna(subset=["fwd12_log_ret", "val_gap", "g20_e10n"]).copy()
    y = (d["fwd12_log_ret"] - d["g20_e10n"]).to_numpy()
    X = d["val_gap"].to_numpy()
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 18})
    return {"lambda": float(model.params[0]), "nw_t": float(model.tvalues[0]),
            "r2": float(model.rsquared), "n": int(len(d))}


# ----------------------------------------------------------------- chart

def make_chart(df, bt0, bt3, summary):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # (1) lambda_used over time (walk-forward estimate)
    ax[0, 0].plot(bt0["date"], bt0["lam_ols"], lw=0.7, alpha=0.5, label="λ OLS (expanding)")
    ax[0, 0].plot(bt0["date"], bt0["lam_used"], lw=1.5, label="λ used (shrunk, clipped)")
    ax[0, 0].axhline(LAMBDA_PRIOR, color="grey", ls=":", label=f"prior {LAMBDA_PRIOR}")
    ax[0, 0].axhline(1.0, color="red", ls="--", lw=0.8, label="full reversion (=1)")
    ax[0, 0].set_title("Walk-forward λ estimate (reversion speed)")
    ax[0, 0].set_ylim(-0.1, 1.1); ax[0, 0].legend(fontsize=8)

    # (2) OOS R² vs RW+drift, per policy (unlagged)
    pols = ["level0_est", "level0_prior", "drift_only", "full_reversion"]
    vals = [summary["unlagged"][p]["oos_r2_vs_rw_drift"] for p in pols]
    colors = ["C0", "C2", "C1", "red"]
    ax[0, 1].bar(pols, vals, color=colors)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_title("OOS R² vs random-walk+drift (>0 = beats baseline)")
    ax[0, 1].tick_params(axis="x", labelrotation=20); ax[0, 1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax[0, 1].text(i, v, f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

    # (3) realized vs level0_est forecast (scatter)
    ax[1, 0].scatter(bt0["fc_level0_est"], bt0["realized"], s=6, alpha=0.3)
    lims = [-1.0, 1.2]
    ax[1, 0].plot(lims, lims, "r-", lw=0.8)
    ax[1, 0].set_xlim(lims); ax[1, 0].set_ylim(lims)
    ax[1, 0].set_xlabel("Level0 forecast (12m log return)")
    ax[1, 0].set_ylabel("realized")
    ax[1, 0].set_title("Level0 forecast vs realized 12m log return")

    # (4) latest fan: point + band over the most recent 25y of origins
    recent = bt0[bt0["date"] >= bt0["date"].max() - pd.DateOffset(years=25)]
    ax[1, 1].plot(recent["date"], recent["realized"], "k.", ms=3, label="realized")
    ax[1, 1].plot(recent["date"], recent["fc_level0_est"], "C0-", lw=1, label="Level0 point")
    ax[1, 1].fill_between(recent["date"], recent["band_q05"], recent["band_q95"],
                          color="C0", alpha=0.15, label="valuation band q05–q95")
    ax[1, 1].axhline(0, color="grey", lw=0.6)
    ax[1, 1].set_title("Level0 point + valuation band vs realized (last 25y origins)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Level 0 structural anchor — walk-forward backtest (PLAN.md §5.1)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "level0_backtest.png", dpi=110)
    print("  -> level0_backtest.png")


# ----------------------------------------------------------------- main

def main():
    print("[1/3] features ...")
    df = build_features()
    feat_cols = ["date", "log_p", "log_cape", "e10n", "g20_e10n", "cape_star_360m",
                 *[f"cape_q{int(q*100):02d}_360m" for q in QS],
                 "cape_pctile_360m", "val_gap", "fwd12_log_ret"]
    df[feat_cols].to_csv(OUT / "level0_features_monthly.csv", index=False)
    print(f"  -> level0_features_monthly.csv {df[feat_cols].shape}, "
          f"val_gap valid from {df.loc[df.val_gap.notna(),'date'].min().date()}")

    print("[2/3] walk-forward backtest (unlagged & lag-3) ...")
    bt0 = backtest(df, lag=0)
    bt3 = backtest(df, lag=LAG)
    summary = {
        "spec": {"H": H, "MED_WIN": MED_WIN, "G_WIN": G_WIN, "MIN_TRAIN": MIN_TRAIN,
                 "lambda_prior": LAMBDA_PRIOR, "lambda_cap": LAMBDA_CAP, "lag_months": LAG},
        "insample_lambda_nw": insample_lambda_nw(df),
        "unlagged": metrics(bt0),
        "lag3_headline": metrics(bt3),
        "honest": honest_metrics(bt0),
    }
    (OUT / "level0_backtest_summary.json").write_text(json.dumps(summary, indent=2))
    print("  -> level0_backtest_summary.json")

    print("[3/3] chart ...")
    make_chart(df, bt0, bt3, summary)

    # console digest
    s = summary["insample_lambda_nw"]
    print(f"\n  in-sample λ = {s['lambda']:.3f} (NW t={s['nw_t']:.2f}), R²={s['r2']:.3f}, n={s['n']}")
    print(f"  backtest origins: {summary['unlagged']['n_origins']} "
          f"({summary['unlagged']['date_range'][0]}..{summary['unlagged']['date_range'][1]})")
    print("\n  policy           RMSE   OOS_R²_vs_RWdrift   (lag-3 headline)")
    for p in ["level0_est", "level0_prior", "drift_only", "full_reversion", "rw_drift", "rw_zero"]:
        u, l = summary["unlagged"][p], summary["lag3_headline"][p]
        print(f"  {p:15s} {u['rmse']:.4f}   {u['oos_r2_vs_rw_drift']:+.4f}            {l['oos_r2_vs_rw_drift']:+.4f}")
    print(f"\n  band coverage: q05–q95 = {summary['unlagged']['band_cover_90']:.1%} (target 90%), "
          f"q25–q75 = {summary['unlagged']['band_cover_50']:.1%} (target 50%)")

    h = summary["honest"]
    print("\n  --- honest framing (audit-driven) ---")
    print(f"  pre-1950 share of total skill: {h['pre1950_share_of_improvement']:.0%}")
    print("  era        level0 vs RWdrift   drift_only vs RWdrift   level0 vs drift_only (valuation増分)")
    for tag in ["full", "post1950", "post1960"]:
        e = h[tag]
        print(f"  {tag:9s}  {e['level0_est_vs_rw_drift']:+.4f}            {e['drift_only_vs_rw_drift']:+.4f}"
              f"                {e['level0_est_vs_drift_only']:+.4f}")


if __name__ == "__main__":
    main()
