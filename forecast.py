#!/usr/bin/env python
"""Phase 6/7 — daily forecast engine (multi-horizon).

Ties the validated ladder together (Level-0 drift + Level-5 GARCH/FHS distribution,
PLAN.md §11/§13) and emits the day's forecast as JSON for the app / daily batch.

Multi-horizon (v2): the same engine now runs at 3/6/12 months (calibrated fans)
and 5/10 years (valuation-based long-run expected return, where CAPE actually
earns its keep — PLAN §A3: R²≈0.43 at 10y vs ~0.04 at 1y). The expensive monthly
volatility models are fit ONCE and reused across every horizon; only the cheap
H-scaled calibration + FHS shape is redone per horizon.

Honesty at long horizons (the project's #1 value): 5y/10y have very few
*independent* (non-overlapping) windows — reported as `n_eff` — so their fans are
labelled "indicative, not calibrated" rather than dressed up as 90% intervals.

Outputs (app/):
  forecast.json   nested {indices→horizons} + a back-compat 1-year top-level mirror
  history.json    past 12m forecasts (annual) with realized outcome + 90%-band hit
Run: uv run python forecast.py   (the GitHub Actions daily job runs exactly this)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase3_level0 import build_features, backtest
from phase5_fanchart import (vol_raw_const, vol_raw_ewma, vol_raw_garch,
                             calib_metrics, fan_from_fhs, QGRID)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
APP = ROOT / "app"

QS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Horizon registry (PLAN §A3 / core-engine design). λ prior rises with H (more of
# the valuation gap closes over a longer horizon: 1−φ^(H/12), φ≈0.92); long horizons
# shrink HARDER toward the prior (shrink_w) because the OLS λ is Stambaugh-biased.
HORIZONS = [
    {"key": "3mo",   "H": 3,   "prior": 0.03, "cap": 0.20, "shrink": 0.5,  "min_cal": 120, "tier": "core"},
    {"key": "6mo",   "H": 6,   "prior": 0.05, "cap": 0.25, "shrink": 0.5,  "min_cal": 120, "tier": "core"},
    {"key": "12mo",  "H": 12,  "prior": 0.10, "cap": 0.35, "shrink": 0.5,  "min_cal": 120, "tier": "core"},
    {"key": "24mo",  "H": 24,  "prior": 0.18, "cap": 0.45, "shrink": 0.5,  "min_cal": 120, "tier": "core"},
    {"key": "36mo",  "H": 36,  "prior": 0.25, "cap": 0.55, "shrink": 0.4,  "min_cal": 96,  "tier": "core"},
    {"key": "60mo",  "H": 60,  "prior": 0.45, "cap": 0.80, "shrink": 0.25, "min_cal": 60,  "tier": "long-run"},
    {"key": "120mo", "H": 120, "prior": 0.65, "cap": 1.00, "shrink": 0.25, "min_cal": 48,  "tier": "long-run"},
    {"key": "180mo", "H": 180, "prior": 0.75, "cap": 1.00, "shrink": 0.25, "min_cal": 40,  "tier": "long-run"},
    {"key": "240mo", "H": 240, "prior": 0.85, "cap": 1.00, "shrink": 0.25, "min_cal": 36,  "tier": "long-run"},
]

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


def block_bootstrap_ci(flags, block, n_boot=2000, lo=0.1, hi=0.9, seed=0):
    """Circular block-bootstrap CI for a coverage rate over OVERLAPPING windows.
    `flags` = 0/1 'realized inside the band' per origin; block ≈ H months absorbs the
    overlap autocorrelation. Returns the [lo, hi] quantiles of the bootstrap coverage
    distribution — the honest error bar on a long-horizon coverage number (few n_eff)."""
    flags = np.asarray(flags, dtype=float)
    nf = len(flags)
    if nf == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(nf / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, nf, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block) % nf for s in starts])[:nf]
        means[b] = flags[idx].mean()
    return [float(np.quantile(means, lo)), float(np.quantile(means, hi))]


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


# --------------------------------------------------- per-horizon walk-forward FHS

def horizon_walk_forward(df, raw_s, spec):
    """Walk-forward FHS predictive quantiles for one horizon, reusing precomputed
    monthly variances `raw_s` (so the costly GARCH fit happens once for all horizons).
    Mirrors phase5_fanchart.predictive_quantiles for the const/ewma/garch models — at
    H=12 it reproduces the validated 1-year output exactly."""
    H = spec["H"]; min_cal = spec["min_cal"]
    bt = backtest(df, lag=0, H=H, lam_prior=spec["prior"], lam_cap=spec["cap"],
                  shrink_w=spec["shrink"]).set_index("date")
    d = bt.dropna(subset=["realized", "fc_level0_est"])
    idx = d.index
    mu = d["fc_level0_est"].to_numpy(); realized = d["realized"].to_numpy()
    n = len(d)
    models = [m for m in ("const", "ewma", "garch", "vix") if m in raw_s]  # vix only when supplied (US)
    sig_raw = {m: np.sqrt(H * raw_s[m].reindex(idx).to_numpy()) for m in models}
    quant = {m: np.full((n, len(QGRID)), np.nan) for m in models}
    sigma_cal = {m: np.full(n, np.nan) for m in models}
    pit = {m: np.full(n, np.nan) for m in models}
    for i in range(n):
        done = np.arange(0, max(i - H + 1, 0))
        if len(done) < min_cal:
            continue
        resid_done = realized[done] - mu[done]; var_done = resid_done ** 2
        for m in models:
            sr_i = sig_raw[m][i]
            if np.isnan(sr_i):
                continue
            base = sig_raw[m][done]; ok = ~np.isnan(base)
            if ok.sum() < min_cal:
                continue
            sigma_cal[m][i] = np.sqrt(np.mean(var_done[ok]) / np.mean(base[ok] ** 2)) * sr_i  # level calibration
        for m in models:
            s12 = sigma_cal[m][i]
            if np.isnan(s12):
                continue
            scal = sigma_cal[m][done]; zok = ~np.isnan(scal) & (scal > 0)
            if zok.sum() < min_cal:
                continue
            z = resid_done[zok] / scal[zok]          # standardize by own σ (self-consistent FHS)
            quant[m][i] = mu[i] + s12 * np.quantile(z, QGRID)
            pit[m][i] = float(np.mean(z <= (realized[i] - mu[i]) / s12))
    return d, mu, realized, quant, sigma_cal, pit


def latest_for_H(df, raw_garch, d, realized, mu, sigma_cal_garch, spec):
    """Live forecast at the last available origin, for horizon H (reuses raw garch)."""
    H = spec["H"]; hf = H / 12.0
    f = df.reset_index(drop=True)
    g = f["g20_e10n"].to_numpy(); vg = f["val_gap"].to_numpy()
    fwd = (f["log_p"].shift(-H) - f["log_p"]).to_numpy()
    L = len(f) - 1
    j = np.arange(0, L + 1)
    ok = ~np.isnan(fwd[j]) & ~np.isnan(vg[j]) & ~np.isnan(g[j]); j = j[ok]
    lam = float(np.dot(vg[j], fwd[j] - g[j] * hf) / np.dot(vg[j], vg[j]))
    lam_used = float(np.clip(spec["shrink"] * lam + (1 - spec["shrink"]) * spec["prior"], 0.0, spec["cap"]))
    mu_L = float(g[L] * hf + lam_used * vg[L])
    sig_raw_L = float(np.sqrt(H * raw_garch.iloc[-1]))
    sr_done = np.sqrt(H * raw_garch.reindex(d.index).to_numpy())
    resid = realized - mu; mok = ~np.isnan(sr_done)
    kfac = float(np.sqrt(np.mean(resid[mok] ** 2) / np.mean(sr_done[mok] ** 2)))
    sigma_L = kfac * sig_raw_L
    cs = sigma_cal_garch; zok = ~np.isnan(cs) & (cs > 0)
    z = resid[zok] / cs[zok]
    P0 = float(np.exp(f["log_p"].iloc[-1]))
    return mu_L, sigma_L, lam_used, z, P0, str(f["date"].iloc[-1].date())


def build_history(d, realized, mu, quant, best_idx_model, P0_series, n_per_year=1):
    """Backfilled answer-key (12m): for ~one origin per year with a known 12m outcome,
    record the forecast median / 50% / 90% bands (price) and the 90%-band hit flag."""
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


def case_studies(d, realized, quant, model, p0_series, studies=CASE_STUDIES):
    """Forecasts the model made at famous turning points + what actually happened —
    the honest record including the misses (e.g. it could not foresee 2008)."""
    dates = d.index
    qmd = int(np.where(np.isclose(QGRID, 0.5))[0][0])
    rows = []
    for tgt, label in studies:
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


JP_TREND_WIN = 240   # 20y rolling window for the price-trend "fair value" (CAPE proxy)
JP_CASE_STUDIES = [
    ("1989-12-31", "Bubble peak (Dec 1989)"),
    ("2003-04-30", "Post-bubble trough (Apr 2003)"),
    ("2009-02-28", "GFC trough (Feb 2009)"),
    ("2012-12-31", "Abenomics launch (Dec 2012)"),
]


def build_features_jp(nk):
    """Japan (Nikkei 225) features. No free long Japanese CAPE exists, so the
    valuation signal is a PRICE-TREND proxy: a point-in-time rolling 20y log-linear
    trend. `val_gap` = trend − log price (>0 = below trend = 'cheap'); `g20_e10n` =
    the trend's annual slope. Reusing the US column names lets the SAME validated
    backtest/anchor run on Japan — λ is estimated walk-forward, so if the trend has
    no predictive power λ shrinks toward ~0 and the drift degrades gracefully to the
    trend's own growth (honest by construction, like Level 0)."""
    mser = nk.set_index("date")["close"].resample("ME").last().dropna()
    df = pd.DataFrame({"date": mser.index, "close": mser.to_numpy()})
    df["log_p"] = np.log(df["close"]); df["r_m"] = df["log_p"].diff()
    lp = df["log_p"].to_numpy(); n = len(df)
    trend = np.full(n, np.nan); slope = np.full(n, np.nan)
    for t in range(n):
        lo = t - JP_TREND_WIN + 1
        if lo < 0:
            continue
        y = lp[lo:t + 1]
        if np.isnan(y).any():
            continue
        x = np.arange(len(y), dtype=float)
        b, a = np.polyfit(x, y, 1)            # slope per month, intercept
        trend[t] = a + b * (len(y) - 1)        # fitted trend level at t
        slope[t] = b
    df["g20_e10n"] = slope * 12                 # annual trend growth (reuse US column name)
    df["val_gap"] = trend - lp                  # below trend = positive = cheap (reversion target)
    df["cape_star_360m"] = np.nan               # no CAPE bands for Japan (FHS supplies the spread)
    for q in [5, 25, 75, 95]:
        df[f"cape_q{q:02d}_360m"] = np.nan
    return df


def japan_indicators(df):
    """A small Japan valuation-context panel (the price-vs-trend gap). Honest scope:
    this is the cruder price-trend proxy used in the drift, not an earnings CAPE."""
    vg = df["val_gap"].dropna()
    if vg.empty:
        return []
    val = float(vg.iloc[-1])
    trail = vg.iloc[-120:] if len(vg) >= 24 else vg
    z = float((val - trail.mean()) / trail.std()) if trail.std() > 0 else 0.0
    pct = float((vg < val).mean())
    return [{"key": "JP_TREND_GAP", "label": "Price vs 20y trend", "value": round(val * 100, 1),
             "z_10y": round(z, 2), "pctile": round(pct, 3), "direction": "bullish",
             "asof": str(df.loc[df["val_gap"].notna(), "date"].iloc[-1].date())}]


def japan_neutral_wf(df, raw_s, spec):
    """Japan walk-forward with a NEUTRAL drift: mu = point-in-time expanding mean of
    completed H-month returns (no valuation timing). This uses MORE history than the
    price-trend anchor (which needs a 240m trend first) and is NOT overfit to the
    recent bull regime — the honest center for Japan. Same GARCH+FHS spread."""
    H = spec["H"]; min_cal = spec["min_cal"]
    fwd = (df["log_p"].shift(-H) - df["log_p"]).to_numpy()
    n = len(df)
    mu_full = np.full(n, np.nan)
    for i in range(n):
        jmax = i - H
        if jmax < 0:
            continue
        vals = fwd[0:jmax + 1]; vals = vals[~np.isnan(vals)]
        if len(vals) >= 120:
            mu_full[i] = vals.mean()
    pos = np.where(~np.isnan(mu_full) & ~np.isnan(fwd))[0]
    if len(pos) == 0:
        return None
    didx = pd.DatetimeIndex(df["date"].iloc[pos])
    mu = mu_full[pos]; realized = fwd[pos]; m_ = len(pos)
    models = ["const", "ewma", "garch"]
    sig_raw = {mm: np.sqrt(H * raw_s[mm].reindex(didx).to_numpy()) for mm in models}
    quant = {mm: np.full((m_, len(QGRID)), np.nan) for mm in models}
    sigma_cal = {mm: np.full(m_, np.nan) for mm in models}
    pit = {mm: np.full(m_, np.nan) for mm in models}
    for i in range(m_):
        done = np.arange(0, max(i - H + 1, 0))
        if len(done) < min_cal:
            continue
        resid_done = realized[done] - mu[done]; var_done = resid_done ** 2
        for mm in models:
            sr_i = sig_raw[mm][i]
            if np.isnan(sr_i):
                continue
            base = sig_raw[mm][done]; ok = ~np.isnan(base)
            if ok.sum() < min_cal:
                continue
            sigma_cal[mm][i] = np.sqrt(np.mean(var_done[ok]) / np.mean(base[ok] ** 2)) * sr_i
        for mm in models:
            s12 = sigma_cal[mm][i]
            if np.isnan(s12):
                continue
            scal = sigma_cal[mm][done]; zok = ~np.isnan(scal) & (scal > 0)
            if zok.sum() < min_cal:
                continue
            z = resid_done[zok] / scal[zok]
            quant[mm][i] = mu[i] + s12 * np.quantile(z, QGRID)
            pit[mm][i] = float(np.mean(z <= (realized[i] - mu[i]) / s12))
    d = pd.DataFrame(index=didx)
    mu_L = float(np.nanmean(fwd))                       # latest expanding-mean drift
    sig_raw_L = float(np.sqrt(H * raw_s["garch"].iloc[-1]))
    sr_done_all = np.sqrt(H * raw_s["garch"].reindex(didx).to_numpy())
    resid = realized - mu; mok = ~np.isnan(sr_done_all)
    sigma_L = float(np.sqrt(np.mean(resid[mok] ** 2) / np.mean(sr_done_all[mok] ** 2))) * sig_raw_L
    cs = sigma_cal["garch"]; zk = ~np.isnan(cs) & (cs > 0); zpool = resid[zk] / cs[zk]
    return d, mu, realized, quant, sigma_cal, pit, mu_L, sigma_L, zpool


def japan_index_block(nk):
    """Nikkei 225 — distribution-only (GARCH+FHS) with a NEUTRAL drift. There is no
    free long Japanese CAPE; a price-trend valuation proxy WAS built and tested
    (build_features_jp) but its walk-forward λ ≈ 0 — no detectable 1-year signal,
    exactly like US CAPE at 1y (PLAN §11) — so it is shown only as CONTEXT, not used
    in the drift. Core horizons only (no honest long-run: too few independent
    windows). Includes the answer-key history + Japanese crisis case studies."""
    df = build_features_jp(nk)            # for the price-trend context indicator
    rm = df["r_m"].to_numpy()
    raw_s = {"const": pd.Series(vol_raw_const(rm), index=df["date"]),
             "ewma": pd.Series(vol_raw_ewma(rm), index=df["date"]),
             "garch": pd.Series(vol_raw_garch(rm), index=df["date"])}
    P0 = float(df["close"].iloc[-1])
    hblocks = {}; hist_records = []
    for spec in [s for s in HORIZONS if s["tier"] == "core" and s["H"] <= 12]:  # Japan history is short; keep its solid range
        H = spec["H"]; key = spec["key"]
        res = japan_neutral_wf(df, raw_s, spec)
        if res is None:
            continue
        d, mu, realized, quant, sigma_cal, pit, mu_L, sigma_L, zpool = res
        valid = {mm: ~np.isnan(quant[mm][:, 0]) for mm in ["const", "ewma", "garch"]}
        common = np.logical_and.reduce([valid[mm] for mm in ["const", "ewma", "garch"]])
        if common.sum() == 0:
            continue
        cov = calib_metrics(realized, quant["garch"], pit["garch"], QGRID, common)
        pit_g = pit["garch"][common]; pit_g = pit_g[~np.isnan(pit_g)]
        pit_counts, _ = np.histogram(pit_g, bins=10, range=(0, 1))
        n_eff = max(1, int(common.sum() / H))
        dates = d.index
        months, bands, q12 = fan_from_fhs(mu_L, sigma_L, zpool, P0, H=H)
        block = {
            "horizon_months": H, "tier": "core",
            "model": {"drift": "neutral drift (historical mean, no valuation timing)", "vol": "garch",
                      "shape": "filtered historical simulation",
                      "mu_log": round(mu_L, 4), "sigma": round(sigma_L, 4)},
            "return_quantiles_pct": {str(q): round((np.exp(mu_L + sigma_L * np.quantile(zpool, q)) - 1) * 100, 1) for q in QS},
            "price_quantiles": {str(q): round(v, 1) for q, v in q12.items()},
            "z_grid": [round(float(z), 4) for z in np.quantile(zpool, QGRID)],  # FHS shape for the probability calc / scenarios
            "fan_path": {"months": months.tolist(),
                         **{f"q{int(q*100):02d}": [round(x, 1) for x in bands[q]] for q in QS}},
            "calibration": {"window": [str(dates[common].min().date()), str(dates[common].max().date())],
                            "n": int(common.sum()), "n_eff": n_eff,
                            "cover50": round(cov["cover_50"], 3), "cover80": round(cov["cover_80"], 3),
                            "cover90": round(cov["cover_90"], 3), "pit_ks": round(cov["pit_ks"], 3),
                            "pit_hist": pit_counts.tolist(),
                            "note": "walk-forward calibrated (~recent window); neutral drift"},
        }
        if key == "12mo":
            p0s = pd.Series(np.exp(df.set_index("date")["log_p"].reindex(d.index).to_numpy()), index=d.index)
            hist_records = build_history(d, realized, mu, {"garch": quant["garch"]}, "garch", p0s)
            block["case_studies"] = case_studies(d, realized, {"garch": quant["garch"]}, "garch", p0s, JP_CASE_STUDIES)
        hblocks[key] = block
        print(f"  [N225 {key:5s}] n={int(common.sum()):4d} n_eff={n_eff:3d}  "
              f"median {(np.exp(mu_L) - 1) * 100:+6.1f}%  cover90={cov['cover_90']:.2f}")
    return {"label": "日経225 / Nikkei 225", "spot": round(P0, 1),
            "indicators": japan_indicators(df), "horizons": hblocks,
            "valuation_note": "price-trend valuation tested: λ≈0, no detectable 1-year signal — shown as context only",
            "_history": hist_records}


def main():
    APP.mkdir(exist_ok=True)
    print("multi-horizon engine: features + monthly vol models (fit once) ...")
    df = build_features().sort_values("date").reset_index(drop=True)
    df["r_m"] = df["log_p"].diff()
    rm = df["r_m"].to_numpy()
    raw_s = {"const": pd.Series(vol_raw_const(rm), index=df["date"]),
             "ewma": pd.Series(vol_raw_ewma(rm), index=df["date"]),
             "garch": pd.Series(vol_raw_garch(rm), index=df["date"])}  # the expensive one, once
    master = pd.read_csv(OUT / "master_monthly.csv", parse_dates=["date"])
    indicators = indicator_panel(master)
    # VIX (S&P 500 option-implied vol, 1990+) as a forward-looking vol model to TEST vs GARCH.
    # VIX is annualized %; (VIX/100)^2/12 is the equivalent monthly variance for the √H machinery.
    _vix = master.set_index("date")["VIXCLS"].reindex(df["date"]).to_numpy()
    raw_s["vix"] = pd.Series((_vix / 100.0) ** 2 / 12.0, index=df["date"])

    blocks = {}
    mirror12 = None
    hist12 = None
    asof = None
    P0_top = None
    print("walk-forward FHS per horizon (reusing the monthly vols) ...")
    for spec in HORIZONS:
        H = spec["H"]; key = spec["key"]
        d, mu, realized, quant, sigma_cal, pit = horizon_walk_forward(df, raw_s, spec)
        valid = {m: ~np.isnan(quant[m][:, 0]) for m in ["const", "ewma", "garch"]}
        common = np.logical_and.reduce([valid[m] for m in ["const", "ewma", "garch"]])
        if common.sum() == 0:
            print(f"  [{key:5s}] no calibrated origins — skipped")
            continue
        cov = calib_metrics(realized, quant["garch"], pit["garch"], QGRID, common)
        pit_g = pit["garch"][common]; pit_g = pit_g[~np.isnan(pit_g)]
        pit_counts, _ = np.histogram(pit_g, bins=10, range=(0, 1))
        n_eff = max(1, int(common.sum() / H))      # ≈ non-overlapping H-month windows
        dates = d.index

        mu_L, sigma_L, lam_used, zpool, P0, asof = latest_for_H(
            df, raw_s["garch"], d, realized, mu, sigma_cal["garch"], spec)
        months, bands, q12 = fan_from_fhs(mu_L, sigma_L, zpool, P0, H=H)
        P0_top = P0

        note = ("walk-forward calibrated; outer bands reliable" if spec["tier"] == "core"
                else f"long-run valuation view: only ~{n_eff} independent {H // 12}y windows — "
                     "treat the band as indicative, not a calibrated 90% interval (Stambaugh-aware)")
        block = {
            "horizon_months": H, "tier": spec["tier"],
            "model": {"drift": "Level-0 structural anchor (log PERxEPS)", "vol": "garch",
                      "shape": "filtered historical simulation",
                      "mu_log": round(mu_L, 4), "sigma": round(sigma_L, 4), "lambda_used": round(lam_used, 3)},
            "return_quantiles_pct": {str(q): round((np.exp(mu_L + sigma_L * np.quantile(zpool, q)) - 1) * 100, 1) for q in QS},
            "price_quantiles": {str(q): round(v, 1) for q, v in q12.items()},
            "z_grid": [round(float(z), 4) for z in np.quantile(zpool, QGRID)],  # FHS shape for the probability calc / scenarios
            "fan_path": {"months": months.tolist(),
                         **{f"q{int(q*100):02d}": [round(x, 1) for x in bands[q]] for q in QS}},
            "calibration": {"window": [str(dates[common].min().date()), str(dates[common].max().date())],
                            "n": int(common.sum()), "n_eff": n_eff,
                            "cover50": round(cov["cover_50"], 3), "cover80": round(cov["cover_80"], 3),
                            "cover90": round(cov["cover_90"], 3), "pit_ks": round(cov["pit_ks"], 3),
                            "pit_hist": pit_counts.tolist(), "note": note},
        }
        if spec["tier"] == "long-run":
            in90 = ((realized >= quant["garch"][:, 0]) & (realized <= quant["garch"][:, -1]))[common]
            ci = block_bootstrap_ci(in90, block=H)
            block["calibration"]["cover90_ci"] = [round(ci[0], 3), round(ci[1], 3)]
            block["long_run"] = {"expected_annualized_pct": round((np.exp(mu_L * 12.0 / H) - 1) * 100, 1),
                                 "method": "CAPE valuation anchor, λ shrunk toward an economic prior (Stambaugh-aware)"}

        if key == "12mo":
            p0_series = pd.Series(np.exp(df.set_index("date")["log_p"].reindex(d.index).to_numpy()), index=d.index)
            block["case_studies"] = case_studies(d, realized, {"garch": quant["garch"]}, "garch", p0_series)
            hist12 = build_history(d, realized, mu, {"garch": quant["garch"]}, "garch", p0_series)
            # back-compat top-level mirror = the validated 1-year flat schema the app already reads
            mirror12 = {
                "model": {"drift": "Level-0 structural anchor (log PERxEPS)", "vol": "garch",
                          "shape": "filtered historical simulation",
                          "mu_12m_log": round(mu_L, 4), "sigma_12m": round(sigma_L, 4)},
                "return_quantiles_pct": block["return_quantiles_pct"],
                "price_quantiles": block["price_quantiles"],
                "fan_path": block["fan_path"],
                "calibration": {"window": block["calibration"]["window"], "n": block["calibration"]["n"],
                                "cover50": block["calibration"]["cover50"], "cover80": block["calibration"]["cover80"],
                                "cover90": block["calibration"]["cover90"], "pit_ks": block["calibration"]["pit_ks"],
                                "pit_hist": block["calibration"]["pit_hist"],
                                "note": "walk-forward calibrated; outer bands reliable"},
                "case_studies": block["case_studies"],
            }
        # VIX-vs-GARCH accuracy test (production stays GARCH; this is an honest diagnostic on the
        # VIX-valid window, 1990+). Adopt VIX only if it clearly wins — reported in the app.
        if "vix" in quant:
            vwin = (~np.isnan(quant["vix"][:, 0])) & valid["garch"]
            if vwin.sum() >= 24:
                cg = calib_metrics(realized, quant["garch"], pit["garch"], QGRID, vwin)
                cvx = calib_metrics(realized, quant["vix"], pit["vix"], QGRID, vwin)
                block["vol_test"] = {
                    "window": [str(dates[vwin].min().date()), str(dates[vwin].max().date())], "n": int(vwin.sum()),
                    "garch_pinball": round(cg["pinball"], 4), "vix_pinball": round(cvx["pinball"], 4),
                    "garch_cover90": round(cg["cover_90"], 3), "vix_cover90": round(cvx["cover_90"], 3),
                    "winner": "garch" if cg["pinball"] <= cvx["pinball"] else "vix"}
        blocks[key] = block
        print(f"  [{key:5s}] n={int(common.sum()):4d} n_eff={n_eff:3d}  "
              f"median {(np.exp(mu_L) - 1) * 100:+6.1f}%  cover90={cov['cover_90']:.2f}  tier={spec['tier']}")

    _last = df.iloc[-1]   # latest valuation building-blocks for the what-if scenarios
    valuation = {"cape": round(float(np.exp(_last["log_cape"])), 1),
                 "cape_star": round(float(np.exp(_last["log_cape"] + _last["val_gap"])), 1),
                 "g_annual": round(float(_last["g20_e10n"]), 4)}
    indices = {"SP500": {"label": "S&P 500", "spot": round(P0_top, 1),
                         "indicators": indicators, "horizons": blocks, "valuation": valuation}}
    nk_path = OUT / "nikkei_daily.csv"
    jp_hist = None
    if nk_path.exists():
        try:
            print("Nikkei 225 (price-trend valuation drift) ...")
            jp = japan_index_block(pd.read_csv(nk_path, parse_dates=["date"]))
            jp_hist = jp.pop("_history", None)
            indices["N225"] = jp
        except Exception as e:  # never let the secondary index break the US product
            print(f"  !! Japan block failed ({e}); shipping US only")

    forecast = {
        "schema_version": 2,
        "asof": asof,
        "default": {"index": "SP500", "horizon": "12mo"},
        "indices": indices,
        # ---- back-compat top-level mirror (the validated 1-year flat schema) ----
        "spot": round(P0_top, 1),
        "horizon_months": 12,
        "model": mirror12["model"],
        "return_quantiles_pct": mirror12["return_quantiles_pct"],
        "price_quantiles": mirror12["price_quantiles"],
        "fan_path": mirror12["fan_path"],
        "indicators": indicators,
        "calibration": mirror12["calibration"],
        "case_studies": mirror12["case_studies"],
    }
    (APP / "forecast.json").write_text(json.dumps(forecast, indent=2))

    us_hit = np.mean([h["in90"] for h in hist12]) if hist12 else float("nan")
    hist_indices = {"SP500": {"n": len(hist12), "hit_rate_90": round(float(us_hit), 3), "records": hist12}}
    if jp_hist:
        jp_hit = np.mean([h["in90"] for h in jp_hist])
        hist_indices["N225"] = {"n": len(jp_hist), "hit_rate_90": round(float(jp_hit), 3), "records": jp_hist}
    (APP / "history.json").write_text(json.dumps(
        {"note": "Backfilled walk-forward answer-key. 'in90' = realized inside the 90% band.",
         "indices": hist_indices,
         # back-compat mirror (S&P 500) for the legacy flat reader
         "n": len(hist12), "hit_rate_90": round(float(us_hit), 3), "records": hist12}, indent=2))

    print(f"\n  asof {asof}: spot {P0_top:,.0f}; horizons {list(blocks.keys())}")
    print(f"  -> app/forecast.json (schema v2: indices→horizons + 1y mirror), app/history.json")


if __name__ == "__main__":
    main()
