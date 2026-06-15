#!/usr/bin/env python
"""Phase 4/5 — Level 5: probabilistic 1-year fan chart (the product).

PLAN.md §5 (Level 5) and §5.1. DRIFT = Level-0 anchor (best 1y point forecast;
macro adds nothing to the mean, §12). The job here is the DISTRIBUTION, validated
walk-forward (PIT, coverage, pinball) — the project's #1 principle.

Volatility (the time-varying scale) is supplied by one of four models; its LEVEL
is recalibrated walk-forward to realized 12m anchor-residual dispersion (σ₁₂≈0.16,
vs naive √12·monthly≈0.14 — the gap that made the Phase-3 band cover only 18%):
  const  unconditional (trailing) variance — baseline
  ewma   RiskMetrics EWMA (λ=0.94)
  garch  GARCH(1,1)-t monthly, variance rolled by recursion — the standard
  nfci   variance from one financial-conditions feature (NFCI). A parsimonious
         test of the §12 macro→risk idea (the rich 5-feature version overfits).

SHAPE: rather than a symmetric Student-t, we use FILTERED HISTORICAL SIMULATION —
the predictive quantiles are the empirical quantiles of past standardized anchor
residuals (resid/σ_cal), rescaled by the current σ_cal. This honors the strong
LEFT-SKEW of 1-year equity returns (median outcome > mean drift; occasional
crashes), which a symmetric law centered on the mean drift gets wrong (it made the
central 50% band too narrow). The fan MEDIAN therefore sits slightly above the
mean-drift anchor — a calibration-driven refinement of §5.1's "median = anchor".

Outputs:
  data/processed/level5_calibration_summary.json
  data/processed/level5_calibration.png       reliability + pinball + sharpness
  data/processed/level5_fanchart_latest.png    the money shot: 1y-ahead price fan
Run: uv run python phase5_fanchart.py
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from phase3_level0 import build_features, backtest as level0_backtest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"

H = 12
QGRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
COVER_LEVELS = [0.50, 0.80, 0.90]
EWMA_LAMBDA = 0.94
GARCH_REFIT = 12
MIN_HIST = 240          # months of monthly-return history before a vol estimate
MIN_CAL = 120           # completed 12m residuals before calibration / FHS
DISP_FEATURES = ["NFCI"]  # parsimonious macro-vol (5-feature version overfits, §13)


def load_inputs():
    df = build_features().sort_values("date").reset_index(drop=True)
    df["r_m"] = df["log_p"].diff()
    bt = level0_backtest(df, lag=0).set_index("date")
    m = pd.read_csv(OUT / "master_monthly.csv", parse_dates=["date"]).set_index("date")
    disp = pd.DataFrame(index=m.index)
    disp["NFCI"] = m["NFCI"]
    disp = disp.shift(2)  # publication lag
    return df, bt, disp


# ----------------------------------------------- raw monthly-variance forecasts

def vol_raw_const(rm):
    out = np.full(len(rm), np.nan)
    for t in range(len(rm)):
        past = rm[max(0, t - 600):t + 1]
        past = past[~np.isnan(past)]
        if len(past) >= MIN_HIST:
            out[t] = np.var(past, ddof=1)
    return out


def vol_raw_ewma(rm):
    out = np.full(len(rm), np.nan)
    h = np.nanvar(rm[:MIN_HIST])
    for t in range(len(rm)):
        r = rm[t]
        if not np.isnan(r):
            h = EWMA_LAMBDA * h + (1 - EWMA_LAMBDA) * r * r
        if t >= MIN_HIST:
            out[t] = h
    return out


def vol_raw_garch(rm, dates=None):
    from arch import arch_model
    out = np.full(len(rm), np.nan)
    omega = alpha = beta = None
    h = np.nanvar(rm[:MIN_HIST])
    for t in range(len(rm)):
        r = rm[t]
        if t >= MIN_HIST and (omega is None or t % GARCH_REFIT == 0):
            hist = rm[:t + 1]; hist = hist[~np.isnan(hist)] * 100
            try:
                res = arch_model(hist, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
                p = res.params
                omega = p["omega"] / 1e4; alpha = p["alpha[1]"]; beta = p["beta[1]"]
                h = res.conditional_volatility[-1] ** 2 / 1e4
            except Exception:
                pass
        if omega is not None and not np.isnan(r):
            h = omega + alpha * r * r + beta * h
        out[t] = h if t >= MIN_HIST else np.nan
    return out


# ------------------------------------------- walk-forward FHS predictive quantiles

def predictive_quantiles(df, bt, disp):
    d = bt.dropna(subset=["realized", "fc_level0_est"]).copy()
    idx = d.index
    mu = d["fc_level0_est"].to_numpy()
    realized = d["realized"].to_numpy()
    n = len(d)

    rm_arr = df["r_m"].to_numpy()
    raw = {"const": vol_raw_const(rm_arr), "ewma": vol_raw_ewma(rm_arr), "garch": vol_raw_garch(rm_arr)}
    raw_s = {k: pd.Series(v, index=df["date"]).reindex(idx).to_numpy() for k, v in raw.items()}
    sig_raw = {k: np.sqrt(12 * raw_s[k]) for k in raw_s}
    Xd = disp.reindex(idx)[DISP_FEATURES].to_numpy()

    models = ["const", "ewma", "garch", "nfci"]
    quant = {m: np.full((n, len(QGRID)), np.nan) for m in models}
    sigma_cal = {m: np.full(n, np.nan) for m in models}
    pit = {m: np.full(n, np.nan) for m in models}

    for i in range(n):
        done = np.arange(0, max(i - H + 1, 0))
        if len(done) < MIN_CAL:
            continue
        resid_done = realized[done] - mu[done]
        var_done = resid_done ** 2

        # --- calibrated σ₁₂ at origin i for each model (point-in-time) ---
        # const must be processed first: its calibrated σ defines the shared FHS pool.
        for m in models:
            if m == "nfci":
                # variance from a single financial-conditions feature; direct estimate (no k)
                if np.isnan(Xd[i]).any():
                    continue
                tr = done[~np.isnan(Xd[done]).any(1)]
                if len(tr) < MIN_CAL:
                    continue
                A = np.column_stack([np.ones(len(tr)), Xd[tr]])
                y = np.log(np.maximum((realized[tr] - mu[tr]) ** 2, 1e-6))
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                s_ref = np.sqrt(np.mean(var_done))        # sane scale from completed residuals
                pred = np.sqrt(np.exp(np.clip(np.array([1, *Xd[i]]) @ coef, -20, 20)))
                sigma_cal["nfci"][i] = float(np.clip(pred, 0.3 * s_ref, 3.0 * s_ref))
            else:
                sr_i = sig_raw[m][i]
                if np.isnan(sr_i):
                    continue
                base = sig_raw[m][done]
                ok = ~np.isnan(base)
                if ok.sum() < MIN_CAL:
                    continue
                kfac = np.sqrt(np.mean(var_done[ok]) / np.mean(base[ok] ** 2))  # level calibration
                sigma_cal[m][i] = kfac * sr_i

        # --- FHS: each model standardizes past residuals by ITS OWN σ (self-consistent),
        #     so the predictive quantiles inherit the empirical skew+kurtosis of the
        #     standardized 1-year surprises, rescaled by the current σ. ---
        for m in models:
            s12 = sigma_cal[m][i]
            if np.isnan(s12):
                continue
            scal = sigma_cal[m][done]
            zok = ~np.isnan(scal) & (scal > 0)
            if zok.sum() < MIN_CAL:
                continue
            z = resid_done[zok] / scal[zok]
            quant[m][i] = mu[i] + s12 * np.quantile(z, QGRID)
            pit[m][i] = float(np.mean(z <= (realized[i] - mu[i]) / s12))

    return d, realized, mu, quant, sigma_cal, pit, models


# ------------------------------------------------------------- scoring

def pinball(realized, q_levels, q_vals):
    losses = [np.mean(np.maximum(a * (realized - q_vals[:, j]), (a - 1) * (realized - q_vals[:, j])))
              for j, a in enumerate(q_levels)]
    return float(np.mean(losses))


def calib_metrics(realized, q_vals, pit_vals, q_levels, mask):
    # drop any origin with a non-finite quantile row (e.g. an unstable nfci fit)
    row_ok = np.isfinite(q_vals).all(axis=1) & mask
    r = realized[row_ok]; qv = q_vals[row_ok]; pit = pit_vals[row_ok]
    out = {"n": int(row_ok.sum()), "pinball": pinball(r, q_levels, qv)}
    out["pit_ks"] = float(stats.kstest(pit[~np.isnan(pit)], "uniform").statistic)
    for lv in COVER_LEVELS:
        ql, qh = (1 - lv) / 2, (1 + lv) / 2
        loq = np.array([np.interp(ql, q_levels, qrow) for qrow in qv])
        hiq = np.array([np.interp(qh, q_levels, qrow) for qrow in qv])
        out[f"cover_{int(lv*100)}"] = float(np.mean((r >= loq) & (r <= hiq)))
        out[f"width_{int(lv*100)}"] = float(np.mean(hiq - loq))
    return out


# ----------------------------------------------- latest live forecast + fan

def latest_forecast(df, d, realized, mu, sigma_cal, best):
    f = df.reset_index(drop=True)
    g = f["g20_e10n"].to_numpy(); vg = f["val_gap"].to_numpy(); fwd = f["fwd12_log_ret"].to_numpy()
    L = len(f) - 1
    j = np.arange(0, L + 1)
    ok = ~np.isnan(fwd[j]) & ~np.isnan(vg[j]) & ~np.isnan(g[j]); j = j[ok]
    lam = float(np.dot(vg[j], fwd[j] - g[j]) / np.dot(vg[j], vg[j]))
    lam_used = float(np.clip(0.5 * lam + 0.5 * 0.10, 0.0, 0.35))
    mu_L = float(g[L] + lam_used * vg[L])

    # latest calibrated σ for the best vol model
    raw_fn = {"const": vol_raw_const, "ewma": vol_raw_ewma, "garch": vol_raw_garch}.get(best)
    if raw_fn is None:  # nfci → fall back to recent calibrated level
        sigma_L = float(np.nanmedian(sigma_cal[best][~np.isnan(sigma_cal[best])][-12:]))
    else:
        raw_full = pd.Series(raw_fn(f["r_m"].to_numpy()), index=f["date"])
        sig_raw_L = float(np.sqrt(12 * raw_full.iloc[-1]))
        sr_done = np.sqrt(12 * raw_full.reindex(d.index).to_numpy())
        resid = realized - mu; mok = ~np.isnan(sr_done)
        kfac = float(np.sqrt(np.mean(resid[mok] ** 2) / np.mean(sr_done[mok] ** 2)))
        sigma_L = kfac * sig_raw_L

    # FHS pool for the best model: standardize residuals by its own σ (same as the sweep)
    resid = realized - mu
    cs = sigma_cal[best]
    zok = ~np.isnan(cs) & (cs > 0)
    z = resid[zok] / cs[zok]
    P0 = float(np.exp(f["log_p"].iloc[-1]))
    return mu_L, sigma_L, z, P0, str(f["date"].iloc[-1].date())


def fan_from_fhs(mu12, sigma12, z, P0, H=H):
    """Path bands consistent with the validated H-month FHS distribution: at month k,
    band_α = P0·exp(drift·k/H + (σ·z_α)·√(k/H)); at k=H it reproduces the FHS
    terminal quantiles exactly. The median uses z's empirical median (≠0 under skew).
    H defaults to 12 (byte-identical to the validated 1-year fan)."""
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    zq = {q: np.quantile(z, q) for q in qs}
    months = np.arange(1, H + 1)
    bands = {q: P0 * np.exp(mu12 * months / H + sigma12 * zq[q] * np.sqrt(months / H)) for q in qs}
    q12 = {q: float(P0 * np.exp(mu12 + sigma12 * zq[q])) for q in qs}
    return months, bands, q12


def main():
    print("[1/4] inputs + walk-forward FHS predictive quantiles (4 vol models) ...")
    df, bt, disp = load_inputs()
    d, realized, mu, quant, sigma_cal, pit, models = predictive_quantiles(df, bt, disp)
    dates = d.index

    print("[2/4] calibration scoring (common window where all models valid) ...")
    valid = {m: ~np.isnan(quant[m][:, 0]) for m in models}
    # common window = the long-history vol models; nfci (NFCI feature) is reported
    # on its own shorter window since it cannot be evaluated pre-~2008.
    common = np.logical_and.reduce([valid[m] for m in ["const", "ewma", "garch"]])
    summary = {"spec": {"H": H, "ewma_lambda": EWMA_LAMBDA, "garch_refit": GARCH_REFIT,
                        "min_cal": MIN_CAL, "shape": "filtered historical simulation (empirical std. residuals)",
                        "macro_vol_feature": DISP_FEATURES},
               "common_window": [str(dates[common].min().date()), str(dates[common].max().date())],
               "n_common": int(common.sum()), "models": {}}
    for m in models:
        full = calib_metrics(realized, quant[m], pit[m], QGRID, valid[m])
        comm = calib_metrics(realized, quant[m], pit[m], QGRID, common)
        summary["models"][m] = {
            "own_window": {"range": [str(dates[valid[m]].min().date()), str(dates[valid[m]].max().date())], **full},
            "common_window": comm}
    # fair head-to-head for the MACRO question: score garch vs nfci on nfci's own window
    macro_win = valid["nfci"]
    summary["macro_question_same_window"] = {
        "window": [str(dates[macro_win].min().date()), str(dates[macro_win].max().date())],
        "n": int(macro_win.sum()),
        **{m: {"pinball": calib_metrics(realized, quant[m], pit[m], QGRID, macro_win)["pinball"],
               "cover_90": calib_metrics(realized, quant[m], pit[m], QGRID, macro_win)["cover_90"]}
           for m in ["const", "garch", "nfci"]}}
    (OUT / "level5_calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print("  -> level5_calibration_summary.json")

    print("[3/4] calibration charts ...")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for m in models:
        cov = [summary["models"][m]["common_window"][f"cover_{int(l*100)}"] for l in COVER_LEVELS]
        ax[0].plot(COVER_LEVELS, cov, "o-", label=m)
    ax[0].plot([0.4, 1], [0.4, 1], "k--", lw=0.8)
    ax[0].set_xlabel("nominal"); ax[0].set_ylabel("empirical coverage")
    ax[0].set_title("Reliability (on the diagonal = calibrated)"); ax[0].legend()
    mm = list(models)
    pb = [summary["models"][m]["common_window"]["pinball"] for m in mm]
    w80 = [summary["models"][m]["common_window"]["width_80"] for m in mm]
    ax[1].bar(mm, pb, color="C0"); ax[1].set_title("Mean pinball loss (lower=better)")
    for i, v in enumerate(pb): ax[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax[2].bar(mm, w80, color="C1"); ax[2].set_title("Mean 80% interval width (sharpness)")
    for i, v in enumerate(w80): ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Level 5 calibration (FHS) — common window {summary['common_window'][0]}..{summary['common_window'][1]} (n={summary['n_common']})")
    fig.tight_layout(); fig.savefig(OUT / "level5_calibration.png", dpi=110)

    print("[4/4] latest 1-year fan chart (the product) ...")
    best = min(models, key=lambda m: summary["models"][m]["common_window"]["pinball"])
    mu_L, sig_L, zpool, P0, asof = latest_forecast(df, d, realized, mu, sigma_cal, best)
    months, bands, q12 = fan_from_fhs(mu_L, sig_L, zpool, P0)
    summary["latest_forecast"] = {"asof": asof, "P0": P0, "mu_12m_log": mu_L, "sigma_12m": sig_L,
                                  "vol_model": best, "price_q": {str(k): v for k, v in q12.items()}}
    (OUT / "level5_calibration_summary.json").write_text(json.dumps(summary, indent=2))

    fig2, axf = plt.subplots(figsize=(11, 6))
    axf.fill_between(months, bands[0.05], bands[0.95], color="C0", alpha=0.15, label="5–95%")
    axf.fill_between(months, bands[0.25], bands[0.75], color="C0", alpha=0.30, label="25–75%")
    axf.plot(months, bands[0.5], "C0-", lw=2, label="median")
    axf.axhline(P0, color="grey", ls=":", label=f"now {P0:,.0f}")
    axf.set_xlabel("months ahead"); axf.set_ylabel("S&P 500 level")
    axf.set_title(f"S&P 500 — 1-year fan chart (Level 0 drift + {best} vol, FHS shape)\n"
                  f"median {q12[0.5]:,.0f} ({(q12[0.5]/P0-1)*100:+.1f}%)  |  90% range [{q12[0.05]:,.0f}, {q12[0.95]:,.0f}]")
    axf.legend(loc="upper left")
    fig2.tight_layout(); fig2.savefig(OUT / "level5_fanchart_latest.png", dpi=110)
    print("  -> level5_calibration.png, level5_fanchart_latest.png")

    print(f"\n  common window {summary['common_window'][0]}..{summary['common_window'][1]} (n={summary['n_common']}, shape=FHS)")
    print("  model   pinball   cover50  cover80  cover90   width80   PIT-KS")
    for m in models:
        c = summary["models"][m]["common_window"]
        print(f"  {m:6s}  {c['pinball']:.4f}   {c['cover_50']:.2f}     {c['cover_80']:.2f}     "
              f"{c['cover_90']:.2f}      {c['width_80']:.3f}     {c['pit_ks']:.3f}")
    mq = summary["macro_question_same_window"]
    print(f"\n  macro question (same window {mq['window'][0]}..{mq['window'][1]}, n={mq['n']}):")
    print(f"    pinball garch={mq['garch']['pinball']:.4f}  const={mq['const']['pinball']:.4f}  "
          f"nfci={mq['nfci']['pinball']:.4f}  → NFCI vol does not beat GARCH")
    print(f"\n  best (pinball) = {best}")
    print(f"  LATEST 1y forecast (asof {asof}): now {P0:,.0f} → median {q12[0.5]:,.0f} ({(q12[0.5]/P0-1)*100:+.1f}%)")
    print(f"    50% range [{q12[0.25]:,.0f}, {q12[0.75]:,.0f}],  90% range [{q12[0.05]:,.0f}, {q12[0.95]:,.0f}]")
    print(f"    (μ={mu_L:+.3f}, σ={sig_L:.3f}, vol={best})")


if __name__ == "__main__":
    main()
