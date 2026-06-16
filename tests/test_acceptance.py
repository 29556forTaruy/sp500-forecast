"""PLAN §5.1 acceptance tests: (1) price identity, (3) degenerate λ, (4) no leakage,
(5) coverage gate. (#2 golden hand-calc lives in test_handcalc.py.)"""
import numpy as np

from phase3_level0 import backtest, H


# ---- #1 identity: P == CAPE * E10n (E10n = P/CAPE by construction) ----
def test_price_identity(features):
    m = features.dropna(subset=["e10n"])
    rel = ((m["P"] - m["CAPE"] * m["e10n"]).abs() / m["P"])
    assert rel.max() < 0.005


# ---- #3 degenerate λ ----
def test_degenerate_lambda_zero(features):
    b = backtest(features, lag=0, lam_prior=0.0, lam_cap=0.0)            # λ→0 ⇒ drift only
    assert np.array_equal(b["fc_level0_est"].to_numpy(), b["fc_drift_only"].to_numpy())


def test_degenerate_lambda_one(features):
    b = backtest(features, lag=0, lam_prior=1.0, lam_cap=1.0, shrink_w=0.0)  # λ→1 ⇒ full reversion
    assert np.array_equal(b["fc_level0_est"].to_numpy(), b["fc_full_reversion"].to_numpy())


# ---- #4 leakage: cut at T+H, the forecast at T is bit-identical (no future peeking) ----
def test_no_future_leakage(features):
    T = "1999-12"
    iT = int(np.where((features["date"].dt.strftime("%Y-%m") == T).to_numpy())[0][0])
    full = backtest(features, lag=0)
    cut = backtest(features.iloc[:iT + H + 1].reset_index(drop=True), lag=0)
    rf = full[full["date"].dt.strftime("%Y-%m") == T]
    rc = cut[cut["date"].dt.strftime("%Y-%m") == T]
    assert len(rf) == 1 and len(rc) == 1, "origin T missing after truncation"
    rf, rc = rf.iloc[0], rc.iloc[0]
    for c in ["fc_level0_est", "lam_ols", "lam_used", "realized",
              "band_q05", "band_q25", "band_q75", "band_q95"]:
        assert rf[c] == rc[c], f"future leakage detected in column {c}"


# ---- #5 coverage gate: the SHIPPED 12m FHS fan covers within [0.86, 0.94] ----
# (uses the calibrated Level-5 cover90 in forecast.json, NOT the uncalibrated Level-0 band)
def test_coverage_gate(forecast_json):
    cov = forecast_json["indices"]["SP500"]["horizons"]["12mo"]["calibration"]["cover90"]
    assert 0.86 <= cov <= 0.94, f"12m cover90={cov} outside calibration gate [0.86, 0.94]"
