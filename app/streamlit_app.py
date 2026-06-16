"""S&P 500 1-year probabilistic forecast — Streamlit app (PLAN.md §8).

Bilingual (English / 日本語, toggle top-right). Reads the daily-batch output
(forecast.json, history.json) produced by `uv run python forecast.py`.
Tabs: ① fan chart, ② indicator heatmap, ③ calibration, ④ answer-key log,
⑤ how it works (methodology). No modeling here — pure presentation.
Run: uv run streamlit run app/streamlit_app.py
"""

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
st.set_page_config(page_title="Market Fan", page_icon="📈", layout="wide")

# App-feel polish (also helps the iOS "Add to Home Screen" PWA experience):
# hide Streamlit's menu/footer chrome and tighten the top padding for mobile.
st.markdown(
    """<style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 2.5rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    fc = json.loads((APP / "forecast.json").read_text())
    hist = json.loads((APP / "history.json").read_text())
    try:
        bench = json.loads((APP / "benchmarks.json").read_text())
    except FileNotFoundError:
        bench = None        # leaderboard not built yet — the ⑧ tab degrades gracefully
    return fc, hist, bench


# ============================================================ i18n strings
# Templates use str.format; numbers are pre-formatted in Python and passed as
# strings so the exact number formatting is identical across both languages.
T = {
    "en": {
        "title": "Market Fan",
        "tagline": "calibrated probability fan charts for equity indices",
        "tab_whatif": "⑦ What-if",
        "tab_bench": "⑧ Benchmark",
        "tab_compare": "⑨ Compare",
        "bench_h": "How does it compare to the world's models?",
        "bench_intro": "Our forecast vs the academic standard benchmarks, walk-forward, scored by "
                       "**OOS R² against the expanding historical mean** (Goyal-Welch 2008's reference). "
                       "Positive = beats the mean. The honest headline: at 1 year almost nothing reliably "
                       "beats the mean and adding CAPE timing barely helps (it even hurts post-1950); "
                       "the model's real edge is slow EPS-growth drift, and a kitchen-sink of all "
                       "predictors overfits and loses. CAPE earns its keep only at long horizons.",
        "bench_point_h": "Point forecast — vs the historical mean",
        "bench_dist_h": "Distribution — vs Gaussian / Student-t / empirical",
        "bench_caption": "At {h}: valuation (CAPE) timing adds **{vi:+.3f}** OOS R² over pure EPS "
                         "drift (**{vi50:+.3f}** post-1950) — {vstory}. **{nb} of {npred}** standard "
                         "predictors beat the mean. This horizon has ~**{neff}** independent windows "
                         "(Stambaugh bias grows at long horizons). Best distribution shape: **{db}**.",
        "bench_vstory_hurts": "i.e. valuation timing does *not* help at this horizon",
        "bench_vstory_helps": "i.e. valuation finally pulls its weight",
        "bench_col_model": "model",
        "bench_col_r2": "OOS R² (full)",
        "bench_col_r2_50": "OOS R² (post-1950)",
        "bench_col_cwp": "Clark-West p",
        "bench_col_dir": "direction hit",
        "bench_col_verdict": "verdict",
        "bench_col_win": "window",
        "bench_col_pinball": "pinball ↓",
        "bench_col_cover": "90% cover",
        "bench_no_hz": "No leaderboard for this horizon — showing the nearest available.",
        "cmp_h": "Compare markets",
        "cmp_intro": "Both markets' 1-year fans, shown as % return from today so the scales line up "
                     "(levels differ: S&P ≈ 7,600 vs Nikkei ≈ 66,000). Wider = more uncertain.",
        "cmp_y": "return from today (%)",
        "pc_h": "Probability calculator",
        "pc_intro": "Read any probability straight off the {h} distribution. Estimates are clamped "
                    "to the modelled 5–95% range (the tails beyond are not claimed).",
        "pc_target": "Target index level",
        "pc_above": "chance of finishing **above {x}** in {h}",
        "pc_below": "chance of finishing **below {x}** in {h}",
        "pc_q_gain": "any gain (> spot)",
        "pc_q_loss10": "a fall > 10%",
        "pc_q_gain10": "a rise > 10%",
        "pc_q_loss20": "a fall > 20%",
        "sc_h": "Scenario — what if?",
        "sc_intro": "Move the sliders to re-draw the {h} fan under your own assumptions. The shape "
                    "(FHS) is kept; only the center, the width, and the starting level change.",
        "sc_cape": "CAPE (valuation)",
        "sc_vol": "Volatility ×",
        "sc_shock": "Immediate price shock %",
        "sc_cape_help": "Where CAPE (the 10-year P/E) starts. Higher = more expensive, so the "
                        "valuation anchor pulls the drift down; lower = cheaper, drift up. "
                        "Default is today's value.",
        "sc_vol_help": "Multiplier on the fan width. 1.0 = the model's current GARCH volatility; "
                       "2.0 = twice as turbulent (wider fan); 0.5 = half (tighter fan).",
        "sc_shock_help": "An instant one-off price move applied today, before the fan is drawn — "
                         "e.g. −20% simulates a crash right now, then forecasts from the new level.",
        "sc_jp_note": "Japan has no CAPE valuation lever (price-only model) — adjust volatility and the immediate shock.",
        "sc_median": "scenario median",
        "sc_range90": "scenario 90% range",
        "sc_base": "base (current forecast)",
        "sc_scenario": "your scenario",
        "caption": "{index} · as of {asof} · spot {spot} · drift = {drift} · vol = {vol} · shape = {shape}",
        "m_spot": "Spot",
        "m_median": "12m median",
        "m_range90": "90% range",
        "m_range50": "50% range",
        "sel_horizon": "Horizon",
        "sel_index": "Market",
        "jp_caveat": (
            "🇯🇵 **Japan model — read this.** There is no free 30-year Japanese CAPE, so a "
            "price-trend valuation proxy was built and **tested** — it shows no detectable 1-year "
            "signal (λ≈0), exactly like US CAPE at 1 year — and is therefore shown only as *context* "
            "(heatmap), not used in the drift. The fan's *width and shape* are calibrated from "
            "history (GARCH + FHS); the *center* is a neutral historical-drift baseline, **not** a "
            "valuation call — it would **not** have called the 1989 bubble top, and the calibration "
            "uses only the recent decades. (JPY, price index.)"),
        "jp_no_panel": "The indicator heatmap uses US macro series (Fed / Shiller) and isn't "
                       "available for the Japan index yet.",
        "jp_us_only": "This view uses the 1-year US walk-forward history and is available for "
                      "S&P 500 only for now.",
        "m_median_h": "{h} median",
        "plain": (
            "**Plain English:** {h} from now the model's *typical* outcome is **{rmed}%** "
            "(≈ {pmed}), with a 1-in-2 chance of landing between **{r25}%** and **{r75}%**, and a "
            "9-in-10 chance between **{r05}%** and **{r95}%**. The median sits above the average "
            "because crashes drag the *mean* down — most periods are modestly up, a few are sharply "
            "down. Valuation (CAPE) is historically extreme, which lowers the *mean* but barely "
            "moves this median; it mainly fattens the downside."),
        "plain_jp": (
            "**Plain English:** {h} from now the model's *typical* outcome is **{rmed}%** "
            "(≈ {pmed}), with a 1-in-2 chance between **{r25}%** and **{r75}%**, and a 9-in-10 "
            "chance between **{r05}%** and **{r95}%** (neutral drift; no valuation tilt — see above)."),
        "plain_long": (
            "**Plain English:** over the next **{h}**, valuation finally does the heavy lifting — "
            "historically CAPE explains a large share of long-run returns (vs almost nothing at 1 "
            "year). The model's central estimate is about **{ann}% per year** ({rmed}% total over "
            "{h}), and today's extreme CAPE is what pulls it down. **Honesty first:** history offers "
            "only ~**{neff}** *independent* {h} windows, so read the band as *indicative*, not a "
            "calibrated 90% interval."),
        "fan_info_long": (
            "Long-run view ({h}): the 90% band historically covered {c90} of outcomes — but with an "
            "**80% bootstrap CI of {ci}**, because this rests on just ~**{neff}** *independent* {h} "
            "windows, far too few to calibrate honestly. Read the median as a valuation-based "
            "expected return and the band as indicative, not a 90% guarantee. This is where CAPE "
            "actually has predictive power (PLAN §A3)."),
        "tabs": ["① Fan chart", "② Indicator heatmap", "③ Calibration",
                 "④ Answer-key log", "⑤ Time machine", "⑥ How it works"],
        # fan chart
        "fan_band_outer": "5–95%",
        "fan_band_inner": "25–75%",
        "fan_median": "median",
        "fan_x": "months ahead",
        "fan_y": "S&P 500 level",
        "fan_now": "now {spot}",
        "fan_info": (
            "Walk-forward calibrated {w0}–{w1} (n={n}): the 80% band historically covered "
            "{c80} and the 90% band {c90} of outcomes (targets 80% / 90%). Drift = Level-0 "
            "structural anchor; width = GARCH conditional volatility; shape = empirical "
            "(filtered historical simulation), so the downside tail is fatter than the upside."),
        # heatmap
        "col_indicator": "Indicator",
        "col_value": "value",
        "col_z": "z (10y)",
        "col_z_help": "Standard deviations from the trailing 10-year mean",
        "col_pctile": "%ile",
        "col_pctile_help": "Rank within full history",
        "col_dir": "high =",
        "col_dir_help": "What a high reading means for equities",
        "col_stance": "stance",
        "col_stance_help": "Signed z: green supports equities, red is a headwind",
        "col_help": "what it is",
        "heat_caption": (
            "`stance` = signed z-score (green = currently supportive of equities, red = headwind). "
            "Hover any cell for detail. CAPE near its all-time-high percentile is the dominant red "
            "flag, but valuation is a *long-horizon* signal — it shapes the distribution more than "
            "the 1-year median."),
        "jp_heat_caption": (
            "`stance` colors the price-vs-trend gap (green = below trend = cheap). This is valuation "
            "*context* only — it was tested and adds no detectable 1-year signal (λ≈0), so it does "
            "not drive the Japan forecast."),
        # calibration
        "cal_h": "Is the model honest?",
        "cal_intro": (
            "A forecast distribution is **calibrated** if outcomes fall inside its bands as often "
            "as claimed. We check this walk-forward (only past data at each date) — the project's "
            "#1 rule."),
        "cal_m50": "50% band covered",
        "cal_m80": "80% band covered",
        "cal_m90": "90% band covered",
        "cal_t50": "target 50%",
        "cal_t80": "target 80%",
        "cal_t90": "target 90%",
        "cal_long_caveat": "⚠️ This is a **long-run ({h})** horizon. The coverage below is computed "
                           "over **overlapping** windows; there are only ~**{neff}** *independent* "
                           "{h} windows in history, so these numbers carry a wide margin of error and "
                           "the PIT histogram is not meaningful at this sample size. Treat the {h} fan "
                           "as a valuation-based view, not a calibrated interval.",
        "vol_test": "**VIX accuracy test:** the fan width was also estimated from VIX "
                    "(option-implied vol) and scored walk-forward against GARCH on {w0}–{w1} "
                    "(n={n}): GARCH pinball **{g}** vs VIX **{v}** (lower = better). {verdict}",
        "vt_garch": "GARCH wins and is kept — VIX is nearly tied at short horizons but, being a "
                    "30-day gauge, degrades at longer ones.",
        "vt_vix": "VIX wins here and would sharpen the fan — flagged for adoption.",
        "pit_x": "PIT bucket (where the outcome fell in the forecast)",
        "pit_y": "count",
        "pit_ideal": "uniform (ideal)",
        "pit_observed": "observed",
        "pit_caption": (
            "**PIT histogram** — if the distribution is perfectly calibrated this is flat (each "
            "bucket ≈ {ideal}). Bars near the edges that are too tall would mean over-confidence "
            "(outcomes hit the tails too often); too short means the bands are too wide. Ours is "
            "close to flat (KS={ks}), mildly imperfect — honest about its own limits."),
        # answer-key
        "ak_h": "Track record (answer-key log)",
        "ak_intro": (
            "{n} past forecasts; the realized index landed inside the 90% band **{hit}** of the "
            "time (target 90%). Kept visible by design (PLAN §0)."),
        "ak_band": "90% band (forecast)",
        "ak_fcmed": "forecast median",
        "ak_realized": "realized (green = in band)",
        "ak_x": "forecast origin",
        "ak_y": "S&P 500 level (1y later)",
        "cs_h": "Crisis case studies — including the misses",
        "cs_intro": "How the model called famous turning points, and what actually happened 12 months later:",
        "cs_in": "✅ in 90% band",
        "cs_out": "❌ outside band (a real miss)",
        "cs_actual": "actual 1y",
        "cs_fcmed": "forecast median {x}%",
        "cs_made": "made: {lo}–{hi} → actual {realized}  \n{verdict}",
        "cs_caption": (
            "The October-2007 row is the honest headline: the model forecast a normal year and the "
            "index fell ~37% into the GFC — **no valuation/vol model foresaw 2008**. The wide bands "
            "and the answer-key exist precisely so the model never pretends otherwise."),
        # methodology
        "mth_h": "How this forecast is built",
        "mth_body": """\
### What this model actually does
It does **not** predict a single number. It outputs a **probability distribution** for where the
S&P 500 could be in 12 months — a most-likely path (the median) plus how far outcomes realistically
spread around it. Two ingredients build that distribution: a **center** (drift) and a
**spread + shape**.

### 1 · The center — a structural "fair value" anchor (Level 0)
Price is just valuation × earnings: `P = PER × EPS`. Taking logs makes that exactly linear, so the
model works in log space:
```
ln P̂(t+12) = ln P_t + g_t + λ · (ln CAPE*_t − ln CAPE_t)
```
In words: **next year's typical level = today's level, carried forward by long-run earnings growth
`g`, and nudged *slightly* toward the 30-year median valuation `CAPE*`.** The nudge strength `λ` is
small — about **0.07–0.11**, estimated from history, not assumed.

Honest caveat: at the **1-year** horizon, valuation (CAPE) adds essentially **no** point-forecast
skill — the useful part of the drift is the earnings-growth term. CAPE earns its keep over the
**long** horizon (7–10 years) and in shaping the **downside**, not in calling next year.

### 2 · Why not "snap back to the average"? (the rejected idea)
The intuitive version — assume an expensive market fully reverts to its median PER within a year
(λ = 1) — was tested and is **catastrophic** (out-of-sample R² = −3.18). It would have predicted
~−40% in early 1997, right before a +31% year. A *small, estimated* λ is the whole point: lean
toward fair value, don't bet the market snaps to it.

### 3 · The spread — how wide is the fan? (GARCH volatility)
The width of the fan is set by **how turbulent markets are right now**. Volatility clusters: calm
begets calm, stress begets stress. A GARCH(1,1) model reads recent volatility and widens or narrows
the fan accordingly. This beats assuming a constant width — a fixed fan can't widen into a crisis,
so it under-covers exactly when it matters.

### 4 · The shape — why the fan isn't symmetric (FHS)
Real 1-year returns are **left-skewed**: crashes are deeper than melt-ups. So instead of a
symmetric bell curve, the model reuses the **actual historical pattern of surprises** (standardized
past residuals), rescaled to today's volatility — "filtered historical simulation." That's why the
**median sits above the mean**, and the **downside tail is fatter than the upside**.

### 5 · What we tried and threw out (on purpose)
We tested 14 macro indicators (yield curve, credit spreads, employment, production, money…) to see
if they sharpen the 1-year point forecast. They **don't** — no statistically detectable signal
(consistent with the academic literature). They carry a *little* information about **risk** (the
spread), but GARCH already captures it. We show this so the model never pretends macro
market-timing works.

### 6 · How we know it's honest — walk-forward calibration
Every number is computed using **only the data available at each past date** (no hindsight). We then
check whether outcomes actually landed inside the bands as often as claimed — across **n ≈ 980
monthly forecasts, 1943–2025**. The 80% and 90% bands cover ≈81% / ≈90%, as they should. The
**answer-key log** and the **2007 miss** are shown deliberately (see the other tabs).

### 7 · Data & updates
Valuation/earnings from **Shiller** (CAPE), macro from the **Fed's FRED**, price from **Yahoo
Finance (^GSPC)**. An automated job refreshes the data and recomputes the forecast **every trading
day**.

### The honest limit
One-year equity returns are mostly **unpredictable**. No valuation or volatility model foresaw 2008.
The value here is an **honest distribution with calibrated uncertainty**, not a crystal ball.
""",
        # time machine (historical replay)
        "tm_h": "Time machine — what the model said, before it knew",
        "tm_intro": "Pick any past date. See the fan the model drew **then** (using only data "
                    "available at the time) and where the index actually landed a year later. "
                    "Green = inside the 90% band, red = an honest miss.",
        "tm_slider": "Pick a past forecast date",
        "tm_need": "Not enough realized history yet for this view.",
        "tm_metric_fc": "forecast median (then)",
        "tm_metric_actual": "actual (1y later)",
        "tm_metric_verdict": "in 90% band?",
        "tm_band90": "90% band (made then)",
        "tm_band50": "50% band",
        "tm_median": "forecast median",
        "tm_real": "actual outcome",
        "tm_x": "months after {origin}",
        "tm_caption_in": "The outcome landed **inside** the band the model drew a year earlier — "
                         "without knowing the future. That is what \"calibrated\" means.",
        "tm_caption_out": "The outcome fell **outside** the band — the model was honestly wrong "
                          "here. These misses are kept visible on purpose (PLAN §0).",
        "tm_caption_gfc": "Early 2008: the model forecast a roughly normal year; the index then "
                          "fell ~37% into the depths of the Global Financial Crisis. **No valuation "
                          "or volatility model foresaw 2008** — the wide bands and this honest log "
                          "exist precisely so the model never pretends otherwise.",
        "pwa_hint": "📱 On iPhone: tap Share → \"Add to Home Screen\" to use this like an app.",
        # table column names (answer-key)
        "tbl": {"origin": "origin", "spot": "spot", "fc_median": "fc_median",
                "fc_lo90": "fc_lo90", "fc_hi90": "fc_hi90", "realized": "realized",
                "realized_ret_pct": "realized %", "in90": "in90"},
    },
    "ja": {
        "title": "マーケット・ファン",
        "tagline": "株価指数の較正済み確率ファンチャート",
        "tab_whatif": "⑦ 試算",
        "tab_bench": "⑧ ベンチマーク",
        "tab_compare": "⑨ 市場比較",
        "bench_h": "世界のモデルと比べてどう?",
        "bench_intro": "うちの予測 vs 学術標準のベンチマークをウォークフォワードで対決。指標は"
                       "**過去平均(拡大窓)に対する OOS R²**(Goyal-Welch 2008 の基準)。プラス=過去平均に勝ち。"
                       "正直な結論:**1年では何もほぼ過去平均に勝てず、CAPEタイミングはほとんど効かない"
                       "(戦後はむしろ害)**。モデルの真の優位は遅いEPS成長ドリフトで、予測子を全部盛ると"
                       "過学習して負ける。CAPEが効くのは長期だけ。",
        "bench_point_h": "点予測 — 過去平均との対決",
        "bench_dist_h": "分布 — 正規 / Student-t / 経験分布との対決",
        "bench_caption": "{h}:バリュエーション(CAPE)タイミングはEPSドリフト単独に対し OOS R² を "
                         "**{vi:+.3f}** 足す(戦後 **{vi50:+.3f}**)— {vstory}。標準予測子は "
                         "**{npred}個中{nb}個**しか過去平均に勝てません。この期間の独立窓は約 **{neff}** 個"
                         "(長期ほどStambaughバイアス大)。分布の最良形状:**{db}**。",
        "bench_vstory_hurts": "=この期間ではバリュエーション・タイミングは効かない",
        "bench_vstory_helps": "=長期でついにバリュエーションが効く",
        "bench_col_model": "モデル",
        "bench_col_r2": "OOS R²(全期間)",
        "bench_col_r2_50": "OOS R²(戦後)",
        "bench_col_cwp": "Clark-West p",
        "bench_col_dir": "方向的中",
        "bench_col_verdict": "判定",
        "bench_col_win": "期間",
        "bench_col_pinball": "pinball ↓",
        "bench_col_cover": "90%カバー",
        "bench_no_hz": "この期間のリーダーボードは無いので、近い期間を表示します。",
        "cmp_h": "市場を比較",
        "cmp_intro": "両市場の1年ファンを、今日からのリターン%で重ねて表示(水準が違うため: "
                     "S&P 約7,600 / 日経 約66,000)。幅が広いほど不確実。",
        "cmp_y": "今日からのリターン(%)",
        "pc_h": "確率計算機",
        "pc_intro": "{h}後の分布から、任意の確率を直接読み取れます。推定はモデルの5〜95%レンジに"
                    "丸めています(その外側の裾は主張しません)。",
        "pc_target": "目標の指数水準",
        "pc_above": "{h}後に **{x} を上回る** 確率",
        "pc_below": "{h}後に **{x} を下回る** 確率",
        "pc_q_gain": "上昇(現値超え)",
        "pc_q_loss10": "10%超の下落",
        "pc_q_gain10": "10%超の上昇",
        "pc_q_loss20": "20%超の下落",
        "sc_h": "シナリオ — もし〜だったら?",
        "sc_intro": "スライダーを動かすと、あなたの前提で{h}ファンを再描画します。形状(FHS)は"
                    "保ったまま、中心・幅・出発点だけが変わります。",
        "sc_cape": "CAPE(バリュエーション)",
        "sc_vol": "ボラティリティ ×",
        "sc_shock": "即時の価格ショック %",
        "sc_cape_help": "出発点の CAPE(10年PER)。高い=割高で、バリュエーション・アンカーが"
                        "ドリフトを押し下げます(低い=割安でドリフト上昇)。既定は現在値。",
        "sc_vol_help": "ファン幅の倍率。1.0=現在のGARCHボラ、2.0=2倍荒れる(幅が広がる)、"
                       "0.5=半分(幅が狭まる)。",
        "sc_shock_help": "ファンを描く前に、今日その場で起きる一回限りの価格変動。"
                         "例:−20%は今クラッシュした想定で、その水準から予測します。",
        "sc_jp_note": "日本にはCAPEバリュエーションのレバーがありません(価格のみのモデル) — ボラと即時ショックで調整してください。",
        "sc_median": "シナリオ中央値",
        "sc_range90": "シナリオ90%レンジ",
        "sc_base": "ベース(現在の予測)",
        "sc_scenario": "あなたのシナリオ",
        "caption": "{index} · {asof} 時点 · 現在値 {spot} · ドリフト = {drift} · ボラ = {vol} · 形状 = {shape}",
        "m_spot": "現在値",
        "m_median": "12ヶ月中央値",
        "m_range90": "90%レンジ",
        "m_range50": "50%レンジ",
        "sel_horizon": "予測期間",
        "sel_index": "市場",
        "jp_caveat": (
            "🇯🇵 **日本版について(お読みください)。** 無料の30年日本版CAPEが無いため、価格トレンドの"
            "バリュエーション代替を作って**検証**しましたが、1年の予測力は検出されず(λ≈0、米国CAPEの1年と同じ)、"
            "**ドリフトには使わず**ヒートマップに*コンテキスト*として表示しています。ファンの*幅と形*は過去から"
            "較正(GARCH+FHS)していますが、*中心*は中立な過去ドリフトのベースラインで、**バリュエーション"
            "判断ではありません** — 特に1989年のバブル天井は**当てません**。較正は近年の窓のみを使っています"
            "(円建て・価格指数)。"),
        "jp_no_panel": "指標ヒートマップは米国のマクロ系列(FRB/Shiller)を使うため、日本指数では"
                       "まだ利用できません。",
        "jp_us_only": "このビューは1年の米国ウォークフォワード履歴を使うため、現状 S&P 500 のみ対応です。",
        "m_median_h": "{h}後の中央値",
        "plain": (
            "**ひとことで言うと:** {h}後のモデルの*典型的な*結果は **{rmed}%**(≈ {pmed})。"
            "2回に1回は **{r25}%〜{r75}%**、10回に9回は **{r05}%〜{r95}%** の範囲に収まる見込みです。"
            "中央値が平均より上にあるのは、暴落が*平均*を押し下げるから — 多くの期間は緩やかな上昇で、"
            "少数の期間に大きく下げます。バリュエーション(CAPE)は歴史的に極端な水準で、これは*平均*を"
            "下げますがこの中央値はほとんど動かさず、主に下振れの裾を太くします。"),
        "plain_jp": (
            "**ひとことで言うと:** {h}後のモデルの*典型的な*結果は **{rmed}%**(≈ {pmed})。"
            "2回に1回は **{r25}%〜{r75}%**、10回に9回は **{r05}%〜{r95}%** の範囲に収まる見込みです"
            "(中立ドリフト・バリュエーション傾斜なし — 上記参照)。"),
        "plain_long": (
            "**ひとことで言うと:** これからの**{h}**では、バリュエーションがついに主役になります — "
            "CAPE は長期リターンの大きな部分を説明します(1年ではほぼ無力)。モデルの中心的な見立ては"
            "**年率 約{ann}%**({h}合計で {rmed}%)で、現在の極端な CAPE がこれを押し下げています。"
            "**正直に言うと:** 歴史上、独立した{h}の窓は約**{neff}**個しかないので、この帯は較正済みの"
            "90%区間ではなく*目安*として見てください。"),
        "fan_info_long": (
            "長期ビュー({h}):過去の90%帯のカバー率は {c90}(**80%ブートストラップ信頼区間 {ci}**)ですが、"
            "これは独立した{h}の窓が約**{neff}**個しかない上での数字で、正直に較正するには少なすぎます。"
            "中央値はバリュエーション根拠の期待リターン、帯は目安として読んでください(90%保証ではありません)。"
            "CAPEが本当に予測力を持つのはこの長期です(PLAN §A3)。"),
        "tabs": ["① ファンチャート", "② 指標ヒートマップ", "③ 較正",
                 "④ 答え合わせログ", "⑤ タイムマシン", "⑥ モデルの仕組み"],
        # fan chart
        "fan_band_outer": "5〜95%",
        "fan_band_inner": "25〜75%",
        "fan_median": "中央値",
        "fan_x": "先の月数",
        "fan_y": "S&P 500 水準",
        "fan_now": "現在 {spot}",
        "fan_info": (
            "ウォークフォワード較正 {w0}〜{w1}(n={n}):過去において80%帯は実際の {c80}、"
            "90%帯は {c90} の結果を含みました(目標 80% / 90%)。ドリフト = Level 0 構造アンカー、"
            "幅 = GARCH 条件付きボラティリティ、形状 = 経験分布(フィルタード・ヒストリカル・"
            "シミュレーション)で、下振れの裾が上振れより太くなっています。"),
        # heatmap
        "col_indicator": "指標",
        "col_value": "値",
        "col_z": "z(10年)",
        "col_z_help": "直近10年平均からの標準偏差(σ)",
        "col_pctile": "%順位",
        "col_pctile_help": "全期間における順位",
        "col_dir": "高い=",
        "col_dir_help": "数値が高いと株式にとって何を意味するか",
        "col_stance": "スタンス",
        "col_stance_help": "符号付きz:緑は株式の追い風、赤は逆風",
        "col_help": "指標の意味",
        "heat_caption": (
            "`スタンス` = 符号付きzスコア(緑 = 現在は株式の追い風、赤 = 逆風)。セルにカーソルを"
            "合わせると詳細が出ます。CAPE が過去最高水準の順位にあることが最大の警戒材料ですが、"
            "バリュエーションは*長期*のシグナルで、1年の中央値よりも分布の形を左右します。"),
        "jp_heat_caption": (
            "`スタンス` は価格とトレンドの乖離を着色(緑=トレンド下=割安)。これはバリュエーションの"
            "*コンテキスト*のみで、検証の結果1年シグナルは検出されず(λ≈0)、日本の予測には使っていません。"),
        # calibration
        "cal_h": "このモデルは正直か?",
        "cal_intro": (
            "予測分布が**較正されている**とは、結果が宣言どおりの頻度でバンド内に収まること。"
            "これをウォークフォワード(各時点で過去データのみ使用)で検証します — 本プロジェクトの"
            "最優先ルールです。"),
        "cal_m50": "50%帯のカバー率",
        "cal_m80": "80%帯のカバー率",
        "cal_m90": "90%帯のカバー率",
        "cal_t50": "目標 50%",
        "cal_t80": "目標 80%",
        "cal_t90": "目標 90%",
        "cal_long_caveat": "⚠️ これは**長期({h})**のホライズンです。下のカバー率は**重複する**窓で"
                           "計算しており、歴史上の独立した{h}の窓は約**{neff}**個しかないため、これらの"
                           "数値は誤差が大きく、この標本サイズでは PIT ヒストグラムも意味を持ちません。"
                           "{h}のファンは較正済み区間ではなく、バリュエーション根拠の見立てとして扱ってください。",
        "vol_test": "**VIX精度テスト:** ファン幅をVIX(オプション予想ボラ)からも推定し、{w0}〜{w1}"
                    "(n={n})でGARCHとウォークフォワード比較: GARCH pinball **{g}** vs VIX **{v}**"
                    "(小さいほど良い)。{verdict}",
        "vt_garch": "GARCHの勝ちで採用継続 — VIXは短期ではほぼ互角ですが、30日指標のため長期で劣化します。",
        "vt_vix": "ここではVIXの勝ちで、ファンを鋭くできます — 採用候補として記録。",
        "pit_x": "PIT バケット(結果が予測分布のどこに落ちたか)",
        "pit_y": "件数",
        "pit_ideal": "一様(理想)",
        "pit_observed": "実測",
        "pit_caption": (
            "**PIT ヒストグラム** — 分布が完全に較正されていれば平ら(各バケット ≈ {ideal})。"
            "端のバーが高すぎれば自信過剰(結果が裾に当たりすぎ)、低すぎればバンドが広すぎ。"
            "本モデルはほぼ平ら(KS={ks})で、わずかに不完全 — 自らの限界に正直です。"),
        # answer-key
        "ak_h": "実績(答え合わせログ)",
        "ak_intro": (
            "過去の予測 {n} 件。実際の指数が90%帯の中に収まったのは **{hit}**(目標90%)。"
            "あえて常に表示しています(計画 §0)。"),
        "ak_band": "90%帯(予測)",
        "ak_fcmed": "予測中央値",
        "ak_realized": "実現値(緑=帯内)",
        "ak_x": "予測時点",
        "ak_y": "S&P 500 水準(1年後)",
        "cs_h": "危機の事例 — 外した例も含めて",
        "cs_intro": "モデルが有名な転換点をどう予測し、12ヶ月後に実際どうなったか:",
        "cs_in": "✅ 90%帯の中",
        "cs_out": "❌ 帯の外(本当の外し)",
        "cs_actual": "実際の1年",
        "cs_fcmed": "予測中央値 {x}%",
        "cs_made": "予測: {lo}〜{hi} → 実際 {realized}  \n{verdict}",
        "cs_caption": (
            "2007年10月の行が正直な見出しです:モデルは平常の年を予測しましたが、指数は金融危機で"
            "約37%下落しました — **どんなバリュエーション/ボラモデルも2008年を予見できませんでした**。"
            "広いバンドと答え合わせログは、モデルが決してそれを取り繕わないために存在します。"),
        # methodology
        "mth_h": "この予測の作り方",
        "mth_body": """\
### このモデルが実際にやっていること
このモデルは**単一の数字を予測しません**。12ヶ月後に S&P 500 がどこにいるかを**確率分布**として
出力します — 最もありそうな経路(中央値)と、その周りに結果が現実的にどれだけ広がるか、です。
分布は2つの要素でできています:**中心(ドリフト)** と **広がり+形状** です。

### 1 · 中心 — 構造的な「適正値」アンカー(Level 0)
株価は「バリュエーション × 利益」にすぎません:`P = PER × EPS`。対数をとるとこれは厳密に一次式に
なるので、モデルは対数空間で動きます:
```
ln P̂(t+12) = ln P_t + g_t + λ · (ln CAPE*_t − ln CAPE_t)
```
言葉にすると:**1年後の典型的な水準 = 今日の水準を、長期の利益成長 `g` で押し上げ、30年中央値
バリュエーション `CAPE*` の方向へ *ほんの少し* 引き戻したもの。** 引き戻しの強さ `λ` は小さく、
歴史から推定して **約 0.07〜0.11**(仮定ではなく実測)です。

正直な注意:**1年**のホライズンでは、バリュエーション(CAPE)は点予測の精度に**ほとんど寄与しません**
— ドリフトで効くのは利益成長の項です。CAPE が本領を発揮するのは**長期(7〜10年)**と、**下振れの形**
であって、来年を当てることではありません。

### 2 · なぜ「平均へ一気に戻す」としないのか(却下した案)
直感的な版 — 割高な市場が1年で中央値 PER へ完全回帰すると仮定する(λ = 1) — を検証したところ、
**壊滅的**でした(アウトオブサンプル R² = −3.18)。これは1997年初に約 −40% を予測しますが、実際は
その後 +31% の年でした。*小さく推定された* λ こそが核心です:適正値へ寄せはするが、市場が一気に
そこへ戻すとは賭けない。

### 3 · 広がり — ファンの幅はどう決まるか(GARCH ボラティリティ)
ファンの幅は **今この瞬間の市場の荒れ具合** で決まります。ボラティリティは固まって出ます:平穏は
平穏を、ストレスはストレスを呼ぶ。GARCH(1,1) モデルが直近のボラを読み、ファンを広げたり狭めたり
します。これは「幅一定」を仮定するより優れています — 固定幅のファンは危機で広がれず、まさに肝心な
ときに裾を取りこぼします。

### 4 · 形状 — なぜファンは左右対称でないのか(FHS)
現実の1年リターンは**左に歪んでいます**:暴落は急騰より深い。だからモデルは左右対称の釣鐘型では
なく、**過去の「サプライズ」の実際のパターン**(標準化した過去残差)を、今日のボラに合わせて
再スケールして使います — これが「フィルタード・ヒストリカル・シミュレーション(FHS)」です。
だから**中央値は平均より上**にあり、**下振れの裾が上振れより太い**のです。

### 5 · 試して捨てたもの(意図的に)
14 のマクロ指標(イールドカーブ、クレジットスプレッド、雇用、生産、マネー…)が1年点予測を鋭くするか
検証しました。**しません** — 統計的に検出できるシグナルはありませんでした(学術研究とも整合)。
これらは**リスク(広がり)**については*わずかに*情報を持ちますが、それも GARCH が既に捉えています。
マクロによる相場タイミングが効くかのように見せないために、これを明示しています。

### 6 · 正直さの担保 — ウォークフォワード較正
すべての数字は、**各過去時点でその時に入手可能だったデータのみ**で計算しています(後知恵なし)。
そのうえで、結果が宣言どおりの頻度で実際にバンド内へ収まったかを検証します — **n ≈ 980 件の
月次予測、1943〜2025年**。80% / 90% 帯は実際に約81% / 約90% をカバーしており、狙いどおりです。
**答え合わせログ**と**2007年の外し**は、あえて表示しています(他タブ参照)。

### 7 · データと更新
バリュエーション/利益は **Shiller**(CAPE)、マクロは **FRB の FRED**、価格は **Yahoo Finance
(^GSPC)**。自動ジョブが**毎営業日**データを更新し、予測を再計算します。

### 正直な限界
1年の株式リターンは大部分が**予測不能**です。どんなバリュエーション/ボラモデルも2008年を予見
できませんでした。ここで価値があるのは**較正された不確実性を伴う正直な分布**であって、水晶玉では
ありません。
""",
        # time machine (historical replay)
        "tm_h": "タイムマシン — 未来を知る前にモデルが言ったこと",
        "tm_intro": "過去の任意の時点を選んでください。モデルが**当時**(その時点で入手可能なデータだけで)"
                    "描いたファンと、1年後に指数が実際どこへ着地したかが見えます。"
                    "緑=90%帯の中、赤=正直な外し。",
        "tm_slider": "過去の予測時点を選ぶ",
        "tm_need": "このビューに必要な実績データがまだ足りません。",
        "tm_metric_fc": "当時の予測中央値",
        "tm_metric_actual": "実際(1年後)",
        "tm_metric_verdict": "90%帯の中?",
        "tm_band90": "90%帯(当時の予測)",
        "tm_band50": "50%帯",
        "tm_median": "予測中央値",
        "tm_real": "実際の結果",
        "tm_x": "{origin} からの経過月数",
        "tm_caption_in": "結果は、1年前にモデルが未来を知らずに描いた帯の**中**に着地しました。"
                         "これが「較正されている」ということです。",
        "tm_caption_out": "結果は帯の**外**に出ました — ここでモデルは正直に外しました。"
                          "こうした外しはあえて見せています(計画 §0)。",
        "tm_caption_gfc": "2008年初:モデルはほぼ平常の年を予測しましたが、指数はその後 金融危機の"
                          "底へ向けて約37%下落しました。**どんなバリュエーション/ボラモデルも"
                          "2008年を予見できませんでした** — 広い帯とこの正直なログは、モデルが"
                          "決してそれを取り繕わないために存在します。",
        "pwa_hint": "📱 iPhoneでは:共有 →「ホーム画面に追加」でアプリのように使えます。",
        # table column names (answer-key)
        "tbl": {"origin": "時点", "spot": "現在値", "fc_median": "予測中央値",
                "fc_lo90": "予測下限(90%)", "fc_hi90": "予測上限(90%)", "realized": "実現値",
                "realized_ret_pct": "実現%", "in90": "帯内"},
    },
}

# values that arrive from forecast.json in English → localized display labels
MODEL_LABEL = {
    "en": {"Level-0 structural anchor (log PERxEPS)": "Level-0 structural anchor (log PER×EPS)",
           "garch": "GARCH", "filtered historical simulation": "filtered historical simulation",
           "neutral drift (historical mean, no valuation timing)": "neutral drift (historical mean, no valuation timing)"},
    "ja": {"Level-0 structural anchor (log PERxEPS)": "Level 0 構造アンカー(対数 PER×EPS)",
           "garch": "GARCH", "filtered historical simulation": "フィルタード・ヒストリカル・シミュレーション",
           "neutral drift (historical mean, no valuation timing)": "中立ドリフト(過去平均・バリュエーション傾斜なし)"},
}
DIRECTION_LABEL = {
    "en": {"bullish": "bullish", "bearish": "bearish", "neutral": "neutral"},
    "ja": {"bullish": "強気", "bearish": "弱気", "neutral": "中立"},
}
IND_LABEL = {
    "shiller_CAPE": {"en": "CAPE (Shiller P/E)", "ja": "CAPE(シラーP/E)"},
    "T10Y2Y": {"en": "Yield curve 10y-2y", "ja": "イールドカーブ(10年−2年)"},
    "DGS10": {"en": "10y Treasury yield", "ja": "米10年国債利回り"},
    "FEDFUNDS": {"en": "Fed funds rate", "ja": "FF金利(政策金利)"},
    "VIXCLS": {"en": "VIX", "ja": "VIX(恐怖指数)"},
    "NFCI": {"en": "Financial conditions (NFCI)", "ja": "金融環境指数(NFCI)"},
    "BAMLH0A0HYM2": {"en": "High-yield credit spread", "ja": "ハイイールド債スプレッド"},
    "UNRATE": {"en": "Unemployment rate", "ja": "失業率"},
    "INDPRO": {"en": "Industrial production (YoY)", "ja": "鉱工業生産(前年比)"},
    "JP_TREND_GAP": {"en": "Price vs 20y trend", "ja": "20年トレンドからの乖離"},
}
IND_HELP = {
    "shiller_CAPE": {
        "en": "Price ÷ 10-year inflation-adjusted earnings. High = expensive = lower long-run returns. The most reliable long-horizon valuation gauge — but weak at 1 year.",
        "ja": "株価 ÷ 過去10年のインフレ調整後利益。高い=割高=長期リターンは低め。最も信頼できる長期バリュエーション指標だが、1年では効きが弱い。"},
    "T10Y2Y": {
        "en": "10-year minus 2-year Treasury yield. Negative (inverted curve) has preceded most recessions; steep/positive = expansion.",
        "ja": "米10年債利回り −2年債利回り。マイナス(逆イールド)は多くの景気後退に先行してきた。急傾斜・プラスは景気拡大。"},
    "DGS10": {
        "en": "10-year Treasury yield. Higher rates pressure valuations but can also signal growth — direction-neutral on its own.",
        "ja": "米10年国債利回り。高金利はバリュエーションの重しになる一方、成長を示すこともあり、それ単独では方向中立。"},
    "FEDFUNDS": {
        "en": "Fed policy rate. Rising = tightening (headwind); falling = easing (tailwind).",
        "ja": "FRB の政策金利。上昇=引き締め(逆風)、低下=緩和(追い風)。"},
    "VIXCLS": {
        "en": "Option-implied 30-day volatility, the 'fear gauge'. Spikes in stress; low = calm (current fan is narrow because VIX is moderate).",
        "ja": "オプションが織り込む30日先のボラティリティ、いわゆる『恐怖指数』。ストレス時に急騰、低い=平穏(現在のファンが狭いのは VIX が中程度のため)。"},
    "NFCI": {
        "en": "Chicago Fed financial-conditions index. Positive = tighter than average financial conditions (headwind).",
        "ja": "シカゴ連銀の金融環境指数。プラス=平均より引き締まった金融環境(逆風)。"},
    "BAMLH0A0HYM2": {
        "en": "High-yield bond spread over Treasuries. Widening = credit stress; tight = risk appetite.",
        "ja": "ハイイールド債と国債の利回り差。拡大=信用不安、縮小=リスク選好。"},
    "UNRATE": {
        "en": "Unemployment rate. A sharp rise (Sahm rule) flags recession.",
        "ja": "失業率。急上昇(サームルール)は景気後退のサイン。"},
    "INDPRO": {
        "en": "Industrial production, year-over-year %. Negative = industrial contraction.",
        "ja": "鉱工業生産の前年比%。マイナス=製造業の縮小。"},
    "JP_TREND_GAP": {
        "en": "How far the Nikkei sits above/below its rolling 20-year price trend (negative = above trend = expensive). A price-only valuation proxy — tested for 1-year predictive power, none detected (λ≈0), so it's shown as context, not used in the forecast drift.",
        "ja": "日経が20年の価格トレンドからどれだけ上/下にあるか(マイナス=トレンド超=割高)。価格のみのバリュエーション代替で、1年予測力を検証した結果シグナルは検出されず(λ≈0)、ドリフトには使わずコンテキストとして表示。"},
}
CASE_LABEL = {
    "Dot-com peak (Dec 1999)": {"en": "Dot-com peak (Dec 1999)", "ja": "ITバブル天井(1999年12月)"},
    "Pre-GFC peak (Oct 2007)": {"en": "Pre-GFC peak (Oct 2007)", "ja": "リーマン前の天井(2007年10月)"},
    "GFC trough (Mar 2009)": {"en": "GFC trough (Mar 2009)", "ja": "金融危機の大底(2009年3月)"},
    "Pre-COVID (Feb 2020)": {"en": "Pre-COVID (Feb 2020)", "ja": "コロナ前(2020年2月)"},
}
HZ_LABEL = {
    "3mo":   {"en": "3 months", "ja": "3か月"},
    "6mo":   {"en": "6 months", "ja": "6か月"},
    "12mo":  {"en": "1 year",   "ja": "1年"},
    "24mo":  {"en": "2 years",  "ja": "2年"},
    "36mo":  {"en": "3 years",  "ja": "3年"},
    "60mo":  {"en": "5 years",  "ja": "5年"},
    "120mo": {"en": "10 years", "ja": "10年"},
    "180mo": {"en": "15 years", "ja": "15年"},
    "240mo": {"en": "20 years", "ja": "20年"},
}
BENCH_MODEL_LABEL = {
    "level0": {"en": "Level-0 (this model)", "ja": "Level 0(本モデル)"},
    "hist_mean": {"en": "Historical mean", "ja": "過去平均"},
    "drift_only": {"en": "EPS-growth drift", "ja": "EPS成長ドリフト"},
    "rw_zero": {"en": "Random walk (=0)", "ja": "ランダムウォーク(=0)"},
    "div_yield": {"en": "Dividend yield (D/P)", "ja": "配当利回り(D/P)"},
    "cape_yield": {"en": "CAPE yield (1/CAPE)", "ja": "CAPE利回り(1/CAPE)"},
    "earn_yield": {"en": "Earnings yield (E/P)", "ja": "益利回り(E/P)"},
    "term_spread": {"en": "Term spread (10y-2y)", "ja": "期間スプレッド(10y-2y)"},
    "goyal_welch": {"en": "Goyal-Welch kitchen sink", "ja": "Goyal-Welch全部入り"},
    "garch_fhs": {"en": "GARCH + FHS (this model)", "ja": "GARCH+FHS(本モデル)"},
    "gaussian": {"en": "Gaussian", "ja": "正規分布"},
    "student_t": {"en": "Student-t", "ja": "Student-t"},
    "historical": {"en": "Empirical (unconditional)", "ja": "経験分布(無条件)"},
}
VERDICT_LABEL = {
    "beats_mean": {"en": "beats mean", "ja": "過去平均に勝ち"},
    "tie": {"en": "tie", "ja": "引き分け"},
    "loses_to_mean": {"en": "loses to mean", "ja": "過去平均に負け"},
    "reference": {"en": "— reference —", "ja": "— 基準 —"},
    "best": {"en": "best", "ja": "最良"},
}

# ============================================================ language toggle
_, lc = st.columns([4, 1])
with lc:
    lang_label = st.segmented_control(
        "Language / 言語", ["English", "日本語"], default="日本語", selection_mode="single")
L = "en" if lang_label == "English" else "ja"


def t(key, **kw):
    s = T[L][key]
    return s.format(**kw) if kw else s


# ============================================================ data + header
fc, hist, bench = load()


def mlabel(v):
    return MODEL_LABEL[L].get(v, v)


def hz_label(k):
    return HZ_LABEL.get(k, {}).get(L, k)


# resolve schema: v2 nests indices→horizons; gracefully fall back to the legacy flat file
if fc.get("schema_version", 1) >= 2:
    indices_obj = fc["indices"]
    idx_default = fc.get("default", {}).get("index") or next(iter(indices_obj))
else:
    indices_obj = {"SP500": {"label": "S&P 500", "spot": fc["spot"],
                             "indicators": fc["indicators"], "horizons": {"12mo": fc}}}
    idx_default = "SP500"

st.title(t("title"))
st.caption(t("tagline"))

# market selector (shown only when >1 index) + horizon selector — both drive every tab
idx_keys = list(indices_obj.keys())
if len(idx_keys) > 1:
    _isel = st.segmented_control(t("sel_index"), [indices_obj[k]["label"] for k in idx_keys],
                                 default=indices_obj[idx_default]["label"], selection_mode="single")
    idx_key = next((k for k in idx_keys if indices_obj[k]["label"] == _isel), idx_default)
else:
    idx_key = idx_default
idx_obj = indices_obj[idx_key]
HORIZONS_AVAIL = idx_obj["horizons"]
spot = idx_obj["spot"]
indicators_data = idx_obj["indicators"]
is_japan = idx_key != "SP500"
# per-index answer-key history + case studies (fall back to the legacy flat mirror)
hist_sel = (hist.get("indices", {}).get(idx_key) if isinstance(hist.get("indices"), dict) else None) or hist
cs_data = (HORIZONS_AVAIL.get("12mo", {}) or {}).get("case_studies") or fc.get("case_studies", [])

hz_keys = list(HORIZONS_AVAIL.keys())
hz_default = "12mo" if "12mo" in hz_keys else hz_keys[0]
if len(hz_keys) > 1:
    _sel = st.segmented_control(t("sel_horizon"), [hz_label(k) for k in hz_keys],
                                default=hz_label(hz_default), selection_mode="single")
    hz = next((k for k in hz_keys if hz_label(k) == _sel), hz_default)
else:
    hz = hz_default
leaf = HORIZONS_AVAIL[hz]
q = leaf["price_quantiles"]; rq = leaf["return_quantiles_pct"]; cal = leaf["calibration"]; m = leaf["model"]
is_long = leaf.get("tier") == "long-run"
hlabel = hz_label(hz)

st.caption(t("caption", index=idx_obj["label"], asof=fc["asof"], spot=f"{spot:,.0f}",
              drift=mlabel(m["drift"]), vol=mlabel(m["vol"]), shape=mlabel(m["shape"])))

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("m_spot"), f"{spot:,.0f}")
c2.metric(t("m_median_h", h=hlabel), f"{q['0.5']:,.0f}", f"{rq['0.5']:+.1f}%")
c3.metric(t("m_range90"), f"{q['0.05']:,.0f} – {q['0.95']:,.0f}")
c4.metric(t("m_range50"), f"{q['0.25']:,.0f} – {q['0.75']:,.0f}")

rkw = dict(rmed=f"{rq['0.5']:+.1f}", pmed=f"{q['0.5']:,.0f}", r25=f"{rq['0.25']:+.1f}",
           r75=f"{rq['0.75']:+.1f}", r05=f"{rq['0.05']:+.1f}", r95=f"{rq['0.95']:+.1f}")
if is_japan:
    st.warning(t("jp_caveat"))
    st.markdown(t("plain_jp", h=hlabel, **rkw))
elif is_long:
    lr = leaf.get("long_run", {})
    st.markdown(t("plain_long", h=hlabel, ann=f"{lr.get('expected_annualized_pct', float('nan')):+.1f}",
                  rmed=f"{rq['0.5']:+.1f}", neff=cal.get("n_eff", "?")))
else:
    st.markdown(t("plain", h=hlabel, **rkw))

_tab_labels = list(t("tabs"))
_tab_labels.append(t("tab_whatif"))                       # ⑦ always
_show_bench = bench is not None
if _show_bench:
    _tab_labels.append(t("tab_bench"))                    # ⑧ when the leaderboard exists
_show_compare = len(indices_obj) > 1
if _show_compare:
    _tab_labels.append(t("tab_compare"))                  # ⑨ when >1 market
_tabs = st.tabs(_tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6 = _tabs[:6]
tab_whatif = _tabs[6]
_nxt = 7
tab_bench = _tabs[_nxt] if _show_bench else None
_nxt += 1 if _show_bench else 0
tab_cmp = _tabs[_nxt] if _show_compare else None

# ----------------------------------------------------------------- fan chart
with tab1:
    fp = leaf["fan_path"]; mo = [0] + fp["months"]
    def path(qk): return [spot] + fp[qk]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mo, y=path("q95"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mo, y=path("q05"), fill="tonexty", fillcolor="rgba(31,119,180,0.13)",
                             line=dict(width=0), name=t("fan_band_outer")))
    fig.add_trace(go.Scatter(x=mo, y=path("q75"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mo, y=path("q25"), fill="tonexty", fillcolor="rgba(31,119,180,0.28)",
                             line=dict(width=0), name=t("fan_band_inner")))
    fig.add_trace(go.Scatter(x=mo, y=path("q50"), line=dict(color="#1f77b4", width=3), name=t("fan_median")))
    fig.add_hline(y=spot, line_dash="dot", line_color="grey", annotation_text=t("fan_now", spot=f"{spot:,.0f}"))
    fig.update_layout(height=480, xaxis_title=t("fan_x"), yaxis_title=t("fan_y"),
                      margin=dict(t=20), hovermode="x unified")
    if is_long:  # multi-year bands span orders of magnitude → log axis keeps them readable
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    if is_long:
        _ci = cal.get("cover90_ci")
        _ci_s = f"{_ci[0]:.0%}–{_ci[1]:.0%}" if _ci and (_ci[1] - _ci[0]) >= 0.02 else "—"
        st.warning(t("fan_info_long", h=hlabel, neff=cal.get("n_eff", "?"),
                     c90=f"{cal['cover90']:.0%}", ci=_ci_s))
    else:
        st.info(t("fan_info", w0=cal["window"][0], w1=cal["window"][1], n=cal["n"],
                  c80=f"{cal['cover80']:.0%}", c90=f"{cal['cover90']:.0%}"))

# ---------------------------------------------------------------- heatmap
with tab2:
  if not indicators_data:
    st.info(t("jp_no_panel"))
  else:
    df = pd.DataFrame(indicators_data)
    def stance(r):
        s = r["z_10y"]
        return -s if r["direction"] == "bearish" else (s if r["direction"] == "bullish" else 0.0)
    df["stance"] = df.apply(stance, axis=1)
    df["label_disp"] = df["key"].map(lambda k: IND_LABEL.get(k, {}).get(L)).fillna(df["label"])
    df["dir_disp"] = df["direction"].map(lambda d: DIRECTION_LABEL[L].get(d, d))
    df["help"] = df["key"].map(lambda k: IND_HELP.get(k, {}).get(L, ""))
    df["percentile"] = (df["pctile"] * 100).round().astype(int)
    show = df[["label_disp", "value", "z_10y", "percentile", "dir_disp", "stance", "help"]]

    def stance_css(v):  # red↔green wash without needing matplotlib (Streamlit Cloud stays light)
        tt = max(-1.0, min(1.0, v / 2.0))
        rgb = (40, 170, 70) if tt >= 0 else (210, 60, 60)
        return f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},{abs(tt)*0.55+0.08:.2f})"

    st.dataframe(
        show.style.map(stance_css, subset=["stance"])
        .format({"value": "{:.2f}", "z_10y": "{:+.2f}", "stance": "{:+.2f}", "percentile": "{:d}%"}),
        use_container_width=True, hide_index=True,
        column_config={
            "label_disp": st.column_config.TextColumn(t("col_indicator")),
            "value": st.column_config.NumberColumn(t("col_value")),
            "z_10y": st.column_config.NumberColumn(t("col_z"), help=t("col_z_help")),
            "percentile": st.column_config.TextColumn(t("col_pctile"), help=t("col_pctile_help")),
            "dir_disp": st.column_config.TextColumn(t("col_dir"), help=t("col_dir_help")),
            "stance": st.column_config.NumberColumn(t("col_stance"), help=t("col_stance_help")),
            "help": st.column_config.TextColumn(t("col_help"), width="large"),
        })
    st.caption(t("jp_heat_caption") if is_japan else t("heat_caption"))

# ---------------------------------------------------------------- calibration
with tab3:
    st.subheader(t("cal_h"))
    st.write(t("cal_intro"))
    if is_long:
        st.warning(t("cal_long_caveat", h=hlabel, neff=cal.get("n_eff", "?")))
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(t("cal_m50"), f"{cal.get('cover50', float('nan')):.0%}", t("cal_t50"))
    cc2.metric(t("cal_m80"), f"{cal['cover80']:.0%}", t("cal_t80"))
    cc3.metric(t("cal_m90"), f"{cal['cover90']:.0%}", t("cal_t90"))
    pit = cal.get("pit_hist", [])
    if pit and not is_long:
        n = sum(pit); k = len(pit); ideal = n / k
        figp = go.Figure()
        figp.add_trace(go.Bar(x=[f"{i*10}–{(i+1)*10}%" for i in range(k)], y=pit, marker_color="#1f77b4",
                              name=t("pit_observed")))
        figp.add_hline(y=ideal, line_dash="dash", line_color="grey", annotation_text=t("pit_ideal"))
        figp.update_layout(height=360, xaxis_title=t("pit_x"), yaxis_title=t("pit_y"),
                           margin=dict(t=20), showlegend=False)
        st.plotly_chart(figp, use_container_width=True)
        st.caption(t("pit_caption", ideal=f"{ideal:.0f}", ks=cal["pit_ks"]))
    vt = leaf.get("vol_test")
    if vt:
        verdict = t("vt_garch") if vt["winner"] == "garch" else t("vt_vix")
        st.caption(t("vol_test", w0=vt["window"][0], w1=vt["window"][1], n=vt["n"],
                     g=vt["garch_pinball"], v=vt["vix_pinball"], verdict=verdict))

# ---------------------------------------------------------------- answer-key
with tab4:
    st.subheader(t("ak_h"))
    h = pd.DataFrame(hist_sel["records"])
    st.write(t("ak_intro", n=hist_sel["n"], hit=f"{hist_sel['hit_rate_90']:.0%}"))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_hi90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_lo90"], fill="tonexty", fillcolor="rgba(31,119,180,0.15)",
                              line=dict(width=0), name=t("ak_band")))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["fc_median"], line=dict(color="#1f77b4", dash="dash"), name=t("ak_fcmed")))
    fig2.add_trace(go.Scatter(x=h["origin"], y=h["realized"], mode="markers",
                              marker=dict(color=["#2ca02c" if v else "#d62728" for v in h["in90"]], size=7),
                              name=t("ak_realized")))
    fig2.update_layout(height=400, xaxis_title=t("ak_x"), yaxis_title=t("ak_y"),
                       margin=dict(t=20), hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader(t("cs_h"))
    st.write(t("cs_intro"))
    cs = pd.DataFrame(cs_data)
    if not cs.empty:
        cols = st.columns(len(cs))
        for col, (_, r) in zip(cols, cs.iterrows()):
            verdict = t("cs_in") if r["in90"] else t("cs_out")
            label_disp = CASE_LABEL.get(r["label"], {}).get(L, r["label"])
            col.markdown(f"**{label_disp}**")
            col.metric(t("cs_actual"), f"{r['realized_ret_pct']:+.0f}%",
                       t("cs_fcmed", x=f"{((r['fc_median']/r['spot'])-1)*100:+.0f}"))
            col.caption(t("cs_made", lo=f"{r['fc_lo90']:,.0f}", hi=f"{r['fc_hi90']:,.0f}",
                          realized=f"{r['realized']:,.0f}", verdict=verdict))
        if not is_japan:
            st.caption(t("cs_caption"))
    cols_order = ["origin", "spot", "fc_median", "fc_lo90", "fc_hi90", "realized", "realized_ret_pct", "in90"]
    st.dataframe(h[cols_order].rename(columns=T[L]["tbl"]).iloc[::-1],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- time machine
with tab5:
    st.subheader(t("tm_h"))
    st.write(t("tm_intro"))
    hm = pd.DataFrame(hist_sel["records"])
    if len(hm) < 2:
        st.info(t("tm_need"))
    else:
        origins = hm["origin"].tolist()
        sel = st.select_slider(t("tm_slider"), options=origins, value=origins[-1])
        r = hm[hm["origin"] == sel].iloc[0]
        spot_v = float(r["spot"]); realized_v = float(r["realized"]); in90 = bool(r["in90"])
        ends = {"q05": float(r["fc_lo90"]), "q25": float(r.get("fc_lo50", r["fc_lo90"])),
                "q50": float(r["fc_median"]), "q75": float(r.get("fc_hi50", r["fc_hi90"])),
                "q95": float(r["fc_hi90"])}
        mths = list(range(0, 13))
        def cone(end): return [spot_v + (end - spot_v) * (mm / 12) ** 0.5 for mm in mths]

        d1, d2, d3 = st.columns(3)
        d1.metric(t("tm_metric_fc"), f"{ends['q50']:,.0f}", f"{(ends['q50']/spot_v-1)*100:+.1f}%")
        d2.metric(t("tm_metric_actual"), f"{realized_v:,.0f}", f"{r['realized_ret_pct']:+.1f}%")
        d3.metric(t("tm_metric_verdict"), "✅" if in90 else "❌")

        figt = go.Figure()
        figt.add_trace(go.Scatter(x=mths, y=cone(ends["q95"]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figt.add_trace(go.Scatter(x=mths, y=cone(ends["q05"]), fill="tonexty", fillcolor="rgba(31,119,180,0.13)",
                                  line=dict(width=0), name=t("tm_band90")))
        figt.add_trace(go.Scatter(x=mths, y=cone(ends["q75"]), line=dict(width=0), showlegend=False, hoverinfo="skip"))
        figt.add_trace(go.Scatter(x=mths, y=cone(ends["q25"]), fill="tonexty", fillcolor="rgba(31,119,180,0.28)",
                                  line=dict(width=0), name=t("tm_band50")))
        figt.add_trace(go.Scatter(x=mths, y=cone(ends["q50"]), line=dict(color="#1f77b4", width=2, dash="dash"),
                                  name=t("tm_median")))
        figt.add_trace(go.Scatter(x=[12], y=[realized_v], mode="markers", name=t("tm_real"),
                                  marker=dict(color="#2ca02c" if in90 else "#d62728", size=15,
                                              symbol="diamond", line=dict(color="white", width=1))))
        figt.add_hline(y=spot_v, line_dash="dot", line_color="grey")
        figt.update_layout(height=420, xaxis_title=t("tm_x", origin=sel), yaxis_title=t("fan_y"),
                           margin=dict(t=20), hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figt, use_container_width=True)

        if not is_japan and not in90 and str(r["origin"]).startswith("2008"):
            st.warning(t("tm_caption_gfc"))   # the Jan-2008 origin fell ~37% into the GFC
        elif in90:
            st.success(t("tm_caption_in"))
        else:
            st.error(t("tm_caption_out"))

# ---------------------------------------------------------------- methodology
with tab6:
    st.subheader(t("mth_h"))
    st.markdown(t("mth_body"))
    st.caption(t("pwa_hint"))

# ---------------------------------------------------------------- what-if (probability calc + scenario)
with tab_whatif:
  zg = leaf.get("z_grid")
  if not zg:
    st.info(t("tm_need"))
  else:
    mu0 = leaf["model"]["mu_log"]; sig0 = leaf["model"]["sigma"]; Hm = leaf["horizon_months"]
    qg = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05..0.95, matches z_grid
    z05, z25, z50, z75, z95 = zg[0], zg[4], zg[9], zg[14], zg[18]

    def interp(xv, xs, ys):
        if xv <= xs[0]: return ys[0]
        if xv >= xs[-1]: return ys[-1]
        for i in range(1, len(xs)):
            if xv <= xs[i]:
                f = (xv - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + f * (ys[i] - ys[i - 1])
        return ys[-1]

    # ---- probability calculator ----
    st.subheader(t("pc_h"))
    st.write(t("pc_intro", h=hlabel))
    pgrid = [spot * math.exp(mu0 + sig0 * z) for z in zg]
    default_x = float(round(spot / 100) * 100)
    x = st.number_input(t("pc_target"), value=default_x, step=float(max(1, round(spot * 0.01))))
    p_below = interp(x, pgrid, qg)
    st.markdown(t("pc_above", x=f"{x:,.0f}", h=hlabel) + f" → **{(1 - p_below):.0%}**  ·  "
                + t("pc_below", x=f"{x:,.0f}", h=hlabel) + f" → **{p_below:.0%}**")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(t("pc_q_gain"), f"{1 - interp(spot, pgrid, qg):.0%}")
    b2.metric(t("pc_q_gain10"), f"{1 - interp(spot * 1.1, pgrid, qg):.0%}")
    b3.metric(t("pc_q_loss10"), f"{interp(spot * 0.9, pgrid, qg):.0%}")
    b4.metric(t("pc_q_loss20"), f"{interp(spot * 0.8, pgrid, qg):.0%}")

    # ---- scenario sliders ----
    st.divider()
    st.subheader(t("sc_h"))
    st.write(t("sc_intro", h=hlabel))
    val = idx_obj.get("valuation"); lam = leaf["model"].get("lambda_used")
    s1, s2, s3 = st.columns(3)
    if val and lam is not None and not is_japan:
        cape_h = s1.slider(t("sc_cape"), 10.0, 45.0, float(val["cape"]), 0.5, help=t("sc_cape_help"))
    else:
        cape_h = None
        s1.caption(t("sc_jp_note"))
    vol_mult = s2.slider(t("sc_vol"), 0.5, 3.0, 1.0, 0.1, help=t("sc_vol_help"))
    shock = s3.slider(t("sc_shock"), -30, 30, 0, 1, help=t("sc_shock_help"))

    if cape_h is not None:
        vg_base = math.log(val["cape_star"]) - math.log(val["cape"])
        growth = mu0 - lam * vg_base                          # back out the (H-scaled) growth drift
        mu_h = growth + lam * (math.log(val["cape_star"]) - math.log(cape_h))
    else:
        mu_h = mu0
    sig_h = sig0 * vol_mult
    spot_h = spot * (1 + shock / 100)
    med0 = spot * math.exp(mu0 + sig0 * z50)
    med_h = spot_h * math.exp(mu_h + sig_h * z50)
    lo_h = spot_h * math.exp(mu_h + sig_h * z05); hi_h = spot_h * math.exp(mu_h + sig_h * z95)
    m1, m2 = st.columns(2)
    m1.metric(t("sc_median"), f"{med_h:,.0f}", f"{(med_h / med0 - 1) * 100:+.1f}% vs base")
    m2.metric(t("sc_range90"), f"{lo_h:,.0f} – {hi_h:,.0f}")

    months = list(range(0, Hm + 1))
    def fan(sp_, mu_, sig_, zq): return [sp_ * math.exp(mu_ * k / Hm + sig_ * zq * math.sqrt(k / Hm)) for k in months]
    figs = go.Figure()
    figs.add_trace(go.Scatter(x=months, y=fan(spot, mu0, sig0, z95), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figs.add_trace(go.Scatter(x=months, y=fan(spot, mu0, sig0, z05), fill="tonexty", fillcolor="rgba(140,140,140,0.12)",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figs.add_trace(go.Scatter(x=months, y=fan(spot, mu0, sig0, z50), line=dict(color="grey", dash="dash"), name=t("sc_base")))
    figs.add_trace(go.Scatter(x=months, y=fan(spot_h, mu_h, sig_h, z95), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    figs.add_trace(go.Scatter(x=months, y=fan(spot_h, mu_h, sig_h, z05), fill="tonexty", fillcolor="rgba(31,119,180,0.16)",
                              line=dict(width=0), name=t("sc_scenario")))
    figs.add_trace(go.Scatter(x=months, y=fan(spot_h, mu_h, sig_h, z50), line=dict(color="#1f77b4", width=2.5), showlegend=False))
    figs.update_layout(height=420, xaxis_title=t("fan_x"), yaxis_title=t("fan_y"),
                       margin=dict(t=20), hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(figs, use_container_width=True)

# ---------------------------------------------------------------- benchmark leaderboard
if tab_bench is not None:
    with tab_bench:
        st.subheader(t("bench_h"))
        st.write(t("bench_intro"))
        bh = bench["indices"]["SP500"]["horizons"]
        bkeys = list(bh.keys())
        bdefault = hz if hz in bh else ("12mo" if "12mo" in bh else bkeys[0])
        if hz not in bh:
            st.caption(t("bench_no_hz"))
        _bsel = st.segmented_control(t("sel_horizon"), [hz_label(k) for k in bkeys],
                                     default=hz_label(bdefault), selection_mode="single")
        bkey = next((k for k in bkeys if hz_label(k) == _bsel), bdefault)
        blk = bh[bkey]; hd = blk["headline"]
        vi = hd.get("valuation_increment") or 0.0; vi50 = hd.get("valuation_increment_post1950")
        vstory = t("bench_vstory_helps") if (vi50 is not None and vi50 > 0.01) else t("bench_vstory_hurts")
        st.info(t("bench_caption", h=hz_label(bkey), vi=vi, vi50=(vi50 if vi50 is not None else 0.0),
                  vstory=vstory, nb=hd.get("n_predictors_beating_mean", "?"), npred=hd.get("n_predictors", "?"),
                  neff=hd.get("n_eff", "?"),
                  db=BENCH_MODEL_LABEL.get(hd.get("dist_best"), {}).get(L, hd.get("dist_best"))))

        st.markdown(f"**{t('bench_point_h')}**")
        prows = [{t("bench_col_model"): BENCH_MODEL_LABEL.get(r["model"], {}).get(L, r["model"]),
                  t("bench_col_r2"): r["oos_r2"], t("bench_col_r2_50"): r.get("oos_r2_post1950"),
                  t("bench_col_cwp"): r.get("cw_p"), t("bench_col_dir"): r["dir_hit"],
                  t("bench_col_verdict"): VERDICT_LABEL.get(r["verdict"], {}).get(L, r["verdict"]),
                  t("bench_col_win"): f"{r['window'][0][:7]}–{r['window'][1][:7]}"} for r in blk["point"]]

        def r2_css(v):
            try:
                vv = float(v)
            except (TypeError, ValueError):
                return ""
            tt = max(-1.0, min(1.0, vv / 0.1)); rgb = (40, 170, 70) if tt >= 0 else (210, 60, 60)
            return f"background-color: rgba({rgb[0]},{rgb[1]},{rgb[2]},{abs(tt)*0.5+0.05:.2f})"

        st.dataframe(pd.DataFrame(prows).style
                     .map(r2_css, subset=[t("bench_col_r2"), t("bench_col_r2_50")])
                     .format({t("bench_col_r2"): "{:+.3f}", t("bench_col_r2_50"): "{:+.3f}",
                              t("bench_col_dir"): "{:.0%}"}, na_rep="—"),
                     use_container_width=True, hide_index=True)

        figb = go.Figure()
        bmodels = [BENCH_MODEL_LABEL.get(r["model"], {}).get(L, r["model"]) for r in blk["point"]]
        figb.add_trace(go.Bar(x=bmodels, y=[r["oos_r2"] for r in blk["point"]],
                              name=t("bench_col_r2"), marker_color="#1f77b4"))
        figb.add_trace(go.Bar(x=bmodels, y=[r.get("oos_r2_post1950") for r in blk["point"]],
                              name=t("bench_col_r2_50"), marker_color="#aec7e8"))
        figb.add_hline(y=0, line_color="grey")
        figb.update_layout(height=380, barmode="group", yaxis_title="OOS R²", margin=dict(t=20),
                           legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(figb, use_container_width=True)

        st.markdown(f"**{t('bench_dist_h')}**")
        drows = [{t("bench_col_model"): BENCH_MODEL_LABEL.get(d["model"], {}).get(L, d["model"]),
                  t("bench_col_pinball"): d["pinball"], t("bench_col_cover"): d["cover90"],
                  "PIT-KS": d["pit_ks"],
                  t("bench_col_verdict"): VERDICT_LABEL.get(d.get("verdict"), {}).get(L, "")} for d in blk["distribution"]]
        st.dataframe(pd.DataFrame(drows).style.format(
            {t("bench_col_pinball"): "{:.4f}", t("bench_col_cover"): "{:.0%}", "PIT-KS": "{:.3f}"}),
            use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- market compare
if tab_cmp is not None:
    with tab_cmp:
        st.subheader(t("cmp_h"))
        st.write(t("cmp_intro"))
        cmp_hz = hz if all(hz in indices_obj[k]["horizons"] for k in indices_obj) else "12mo"
        cmp_style = {"SP500": ("#1f77b4", "rgba(31,119,180,0.13)"),
                     "N225": ("#d62728", "rgba(214,39,40,0.13)")}
        def rpath(fp, s, qk): return [0.0] + [(v / s - 1) * 100 for v in fp[qk]]
        figc = go.Figure()
        for k, obj in indices_obj.items():
            if cmp_hz not in obj["horizons"]:
                continue
            fpc = obj["horizons"][cmp_hz]["fan_path"]; sp = obj["spot"]; mo = [0] + fpc["months"]
            line_c, fill_c = cmp_style.get(k, ("#888888", "rgba(136,136,136,0.13)"))
            figc.add_trace(go.Scatter(x=mo, y=rpath(fpc, sp, "q95"), line=dict(width=0), showlegend=False, hoverinfo="skip"))
            figc.add_trace(go.Scatter(x=mo, y=rpath(fpc, sp, "q05"), fill="tonexty", fillcolor=fill_c,
                                      line=dict(width=0), name=f"{obj['label']} · {hz_label(cmp_hz)} (90%)"))
            figc.add_trace(go.Scatter(x=mo, y=rpath(fpc, sp, "q50"), line=dict(color=line_c, width=2.5), showlegend=False))
        figc.add_hline(y=0, line_dash="dot", line_color="grey")
        figc.update_layout(height=460, xaxis_title=t("fan_x"), yaxis_title=t("cmp_y"),
                           margin=dict(t=20), hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figc, use_container_width=True)
