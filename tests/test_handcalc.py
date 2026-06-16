"""PLAN §5.1 acceptance #2 — golden hand-calc fixture, matched in log space to 1e-9.

At 1997-01 the Level-0 building blocks are (hand-verified against the data):
    g20      =  0.054596   (20y real E10 growth)
    val_gap  = -0.635474   (ln CAPE* - ln CAPE; CAPE ~28 >> 30y median → expensive)
    lambda   =  0.084136   (OLS 0.068271, shrunk 50/50 to the 0.10 prior, clipped)
    ln P̂/P  = g20 + lambda*val_gap = 0.054596 + 0.084136*(-0.635474) = 0.0011299  (≈ +0.11%)
This pins the famous 1997 case: a *small* λ forecasts ~flat (the market then rose +31%),
whereas the rejected λ=1 would have predicted ≈ -40%. A break in the feature math or the λ
policy moves this number and fails the test."""
import math

from conftest import level0_point_logret

GOLDEN_1997_01_LOG_RET = 0.001129871621140574


def test_golden_handcalc_1997(features):
    got = level0_point_logret(features, "1997-01")
    assert math.isclose(got, GOLDEN_1997_01_LOG_RET, abs_tol=1e-9), \
        f"hand-calc drift: got {got!r}, expected {GOLDEN_1997_01_LOG_RET!r}"
    # arithmetic sanity: the components reconcile to the total (hand-checkable)
    assert abs((0.054596 + 0.084136 * -0.635474) - got) < 1e-4
