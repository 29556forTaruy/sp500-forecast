"""PLAN §5.1 episode regression tests: at six historical turning points the Level-0
12-month forecast must stay within [-35%, +60%] (catches the catastrophic λ=1 regression
and any feature-math break)."""
import numpy as np
import pytest

from conftest import level0_point_logret

EPISODES = [
    ("1921-08", "post-WWI recession"),
    ("1974-12", "stagflation trough"),
    ("1997-01", "pre-Asian crisis"),
    ("1999-12", "dot-com peak"),
    ("2009-03", "GFC trough"),
    ("2021-01", "post-COVID low"),
]


@pytest.mark.parametrize("ym,label", EPISODES)
def test_episode_forecast_in_range(features, ym, label):
    ret = np.exp(level0_point_logret(features, ym)) - 1.0
    assert -0.35 <= ret <= 0.60, f"{ym} ({label}): forecast {ret:+.1%} outside [-35%, +60%]"
