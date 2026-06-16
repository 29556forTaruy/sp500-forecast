"""Shared pytest fixtures + the production point-forecast helper (PLAN §5.1 tests).

Tests run on the git-committed processed CSVs (data/processed/shiller_monthly.csv)
and the committed app/forecast.json — NO data refetch needed.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3_level0 import (build_features, backtest, estimate_lambda,  # noqa: E402
                          H, MIN_TRAIN, LAMBDA_PRIOR, LAMBDA_CAP)


@pytest.fixture(scope="session")
def features():
    return build_features().reset_index(drop=True)


@pytest.fixture(scope="session")
def bt_default(features):
    return backtest(features, lag=0)


@pytest.fixture(scope="session")
def forecast_json():
    return json.loads((ROOT / "app" / "forecast.json").read_text())


def level0_point_logret(features, ym):
    """Production Level-0 12-month point forecast (log return) at month `ym` (YYYY-MM),
    using the SAME expanding-window λ policy as phase3_level0.backtest(): shrink the OLS
    slope 50/50 toward the prior and clip to [0, LAMBDA_CAP]. When fewer than MIN_TRAIN
    completed (val_gap, fwd) pairs exist (e.g. 1921-08, which has no walk-forward origin),
    it correctly falls back to the prior λ — this is itself a no-leakage property."""
    df = features
    mask = (df["date"].dt.strftime("%Y-%m") == ym).to_numpy()
    assert mask.any(), f"{ym} not present in features"
    i = int(np.where(mask)[0][0])
    g = df["g20_e10n"].to_numpy(); vg = df["val_gap"].to_numpy()
    fwd = (df["log_p"].shift(-H) - df["log_p"]).to_numpy()
    lam = LAMBDA_PRIOR
    jmax = i - H
    if jmax >= 0:
        j = np.arange(0, jmax + 1)
        ok = ~np.isnan(fwd[j]) & ~np.isnan(vg[j]) & ~np.isnan(g[j])
        j = j[ok]
        if len(j) >= MIN_TRAIN:
            lam_ols = estimate_lambda(fwd[j] - g[j], vg[j])  # H=12 → hf=1, matches backtest
            lam = float(np.clip(0.5 * lam_ols + 0.5 * LAMBDA_PRIOR, 0.0, LAMBDA_CAP))
    return float(g[i] + lam * vg[i])
