#!/usr/bin/env python3
"""Weekly digest → Discord webhook (stdlib only — no deps, runs on a bare runner).

Reads app/forecast.json (produced by the daily job) and posts a short S&P 500
summary to a Discord channel. Configure once:
  • create a Discord channel webhook, copy its URL
  • add it as a GitHub Actions secret named DISCORD_WEBHOOK_URL
  • (optional) set repo variable APP_URL to your Streamlit app link
If DISCORD_WEBHOOK_URL is unset, this just PRINTS the message (safe to run locally
or before the secret exists). The channel can be swapped (LINE/Slack/email) by
changing only the post() call — build_message() is channel-agnostic.

Run: python3 digest.py
"""

import json
import os
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent / "app"
DEFAULT_APP_URL = "https://sp500-forecast-4xldbbjrzkvbbxpcmhblsp.streamlit.app"


def build_message() -> str:
    fc = json.loads((APP / "forecast.json").read_text())
    asof = fc.get("asof", "?")
    # 1-year S&P 500 leaf (works on schema v2 or the legacy flat file)
    if fc.get("schema_version", 1) >= 2:
        sp = fc["indices"]["SP500"]
        spot = sp["spot"]; h = sp["horizons"]["12mo"]; inds = sp["indicators"]
    else:
        spot = fc["spot"]; h = fc; inds = fc.get("indicators", [])
    q = h["price_quantiles"]; rq = h["return_quantiles_pct"]; cal = h["calibration"]
    cape = next((i for i in inds if i["key"] == "shiller_CAPE"), None)
    app_url = os.environ.get("APP_URL", DEFAULT_APP_URL)

    lines = [
        f"📈 **S&P 500 — 1-year forecast** (as of {asof})",
        f"Spot **{spot:,.0f}** → 12m median **{q['0.5']:,.0f}** ({rq['0.5']:+.1f}%)",
        f"50% range {q['0.25']:,.0f}–{q['0.75']:,.0f}  ·  90% range {q['0.05']:,.0f}–{q['0.95']:,.0f}",
    ]
    if cape:
        lines.append(f"CAPE **{cape['value']:.0f}** ({cape['pctile'] * 100:.0f}th %ile) · "
                     f"walk-forward 90% coverage {cal['cover90']:.0%} (n={cal['n']})")
    # 10-year long-run view, if present (valuation-based expected return)
    lr = (fc.get("indices", {}).get("SP500", {}).get("horizons", {}).get("120mo", {}) or {})
    if lr.get("long_run"):
        lines.append(f"10y valuation view: ~{lr['long_run']['expected_annualized_pct']:+.1f}%/yr "
                     f"(indicative, n_eff={lr['calibration'].get('n_eff', '?')})")
    lines.append(app_url)
    return "\n".join(lines)


def post(msg: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("DISCORD_WEBHOOK_URL not set — message preview:\n")
        print(msg)
        return
    data = json.dumps({"content": msg}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"posted to Discord (HTTP {r.status})")


if __name__ == "__main__":
    post(build_message())
