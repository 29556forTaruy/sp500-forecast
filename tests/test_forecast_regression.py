"""Structural-invariant regression on the shipped app/forecast.json (formalizes the
informal golden check). Asserts schema invariants rather than exact daily values, so a
legitimate daily data refresh does not red the build."""


def test_schema_and_mirror(forecast_json):
    fc = forecast_json
    assert fc.get("schema_version") == 2
    # back-compat top-level 1-year mirror must exist and match the nested 12mo block
    sp = fc["indices"]["SP500"]
    h12 = sp["horizons"]["12mo"]
    for k in ["spot", "model", "return_quantiles_pct", "price_quantiles", "indicators",
              "calibration", "case_studies"]:
        assert k in fc, f"top-level mirror missing {k}"
    assert fc["price_quantiles"] == h12["price_quantiles"], "mirror diverged from nested 12mo"
    assert fc["spot"] == sp["spot"]


def test_quantiles_monotone(forecast_json):
    for hz, blk in forecast_json["indices"]["SP500"]["horizons"].items():
        pq = blk["price_quantiles"]
        vals = [pq[k] for k in ["0.05", "0.25", "0.5", "0.75", "0.95"]]
        assert vals == sorted(vals), f"{hz} price quantiles not monotone: {vals}"
        assert vals[0] > 0, f"{hz} has a non-positive price quantile"


def test_both_indices_present(forecast_json):
    idx = forecast_json["indices"]
    assert "SP500" in idx and "N225" in idx
    assert idx["N225"]["horizons"], "Nikkei has no horizons"
