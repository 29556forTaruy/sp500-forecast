"""S&P 500 1-year probabilistic forecast — Streamlit app (PLAN.md §8).

Bilingual (English / 日本語, toggle top-right). Reads the daily-batch output
(forecast.json, history.json) produced by `uv run python forecast.py`.
Tabs: ① fan chart, ② indicator heatmap, ③ calibration, ④ answer-key log,
⑤ how it works (methodology). No modeling here — pure presentation.
Run: uv run streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
st.set_page_config(page_title="S&P 500 forecast / 予測", page_icon="📈", layout="wide")


@st.cache_data
def load():
    fc = json.loads((APP / "forecast.json").read_text())
    hist = json.loads((APP / "history.json").read_text())
    return fc, hist


# ============================================================ i18n strings
# Templates use str.format; numbers are pre-formatted in Python and passed as
# strings so the exact number formatting is identical across both languages.
T = {
    "en": {
        "title": "S&P 500 — 1-year probabilistic forecast",
        "caption": "As of {asof} · spot {spot} · drift = {drift} · vol = {vol} · shape = {shape}",
        "m_spot": "Spot",
        "m_median": "12m median",
        "m_range90": "90% range",
        "m_range50": "50% range",
        "plain": (
            "**Plain English:** a year from now the model's *typical* outcome is **{rmed}%** "
            "(≈ {pmed}), with a 1-in-2 chance of landing between **{r25}%** and **{r75}%**, and a "
            "9-in-10 chance between **{r05}%** and **{r95}%**. The median sits above the average "
            "because crashes drag the *mean* down — most years are modestly up, a few are sharply "
            "down. Valuation (CAPE) is historically extreme, which lowers the *mean* but barely "
            "moves the 1-year median; it mainly fattens the downside."),
        "tabs": ["① Fan chart", "② Indicator heatmap", "③ Calibration",
                 "④ Answer-key log", "⑤ How it works"],
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
        # table column names (answer-key)
        "tbl": {"origin": "origin", "spot": "spot", "fc_median": "fc_median",
                "fc_lo90": "fc_lo90", "fc_hi90": "fc_hi90", "realized": "realized",
                "realized_ret_pct": "realized %", "in90": "in90"},
    },
    "ja": {
        "title": "S&P 500 — 1年先の確率予測",
        "caption": "{asof} 時点 · 現在値 {spot} · ドリフト = {drift} · ボラ = {vol} · 形状 = {shape}",
        "m_spot": "現在値",
        "m_median": "12ヶ月中央値",
        "m_range90": "90%レンジ",
        "m_range50": "50%レンジ",
        "plain": (
            "**ひとことで言うと:** 1年後のモデルの*典型的な*結果は **{rmed}%**(≈ {pmed})。"
            "2回に1回は **{r25}%〜{r75}%**、10回に9回は **{r05}%〜{r95}%** の範囲に収まる見込みです。"
            "中央値が平均より上にあるのは、暴落が*平均*を押し下げるから — 多くの年は緩やかな上昇で、"
            "少数の年に大きく下げます。バリュエーション(CAPE)は歴史的に極端な水準で、これは*平均*を"
            "下げますが1年の中央値はほとんど動かさず、主に下振れの裾を太くします。"),
        "tabs": ["① ファンチャート", "② 指標ヒートマップ", "③ 較正",
                 "④ 答え合わせログ", "⑤ モデルの仕組み"],
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
        # table column names (answer-key)
        "tbl": {"origin": "時点", "spot": "現在値", "fc_median": "予測中央値",
                "fc_lo90": "予測下限(90%)", "fc_hi90": "予測上限(90%)", "realized": "実現値",
                "realized_ret_pct": "実現%", "in90": "帯内"},
    },
}

# values that arrive from forecast.json in English → localized display labels
MODEL_LABEL = {
    "en": {"Level-0 structural anchor (log PERxEPS)": "Level-0 structural anchor (log PER×EPS)",
           "garch": "GARCH", "filtered historical simulation": "filtered historical simulation"},
    "ja": {"Level-0 structural anchor (log PERxEPS)": "Level 0 構造アンカー(対数 PER×EPS)",
           "garch": "GARCH", "filtered historical simulation": "フィルタード・ヒストリカル・シミュレーション"},
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
}
CASE_LABEL = {
    "Dot-com peak (Dec 1999)": {"en": "Dot-com peak (Dec 1999)", "ja": "ITバブル天井(1999年12月)"},
    "Pre-GFC peak (Oct 2007)": {"en": "Pre-GFC peak (Oct 2007)", "ja": "リーマン前の天井(2007年10月)"},
    "GFC trough (Mar 2009)": {"en": "GFC trough (Mar 2009)", "ja": "金融危機の大底(2009年3月)"},
    "Pre-COVID (Feb 2020)": {"en": "Pre-COVID (Feb 2020)", "ja": "コロナ前(2020年2月)"},
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
fc, hist = load()
spot = fc["spot"]; q = fc["price_quantiles"]; rq = fc["return_quantiles_pct"]; cal = fc["calibration"]
m = fc["model"]


def mlabel(v):
    return MODEL_LABEL[L].get(v, v)


st.title(t("title"))
st.caption(t("caption", asof=fc["asof"], spot=f"{spot:,.0f}",
              drift=mlabel(m["drift"]), vol=mlabel(m["vol"]), shape=mlabel(m["shape"])))

c1, c2, c3, c4 = st.columns(4)
c1.metric(t("m_spot"), f"{spot:,.0f}")
c2.metric(t("m_median"), f"{q['0.5']:,.0f}", f"{rq['0.5']:+.1f}%")
c3.metric(t("m_range90"), f"{q['0.05']:,.0f} – {q['0.95']:,.0f}")
c4.metric(t("m_range50"), f"{q['0.25']:,.0f} – {q['0.75']:,.0f}")

st.markdown(t("plain",
              rmed=f"{rq['0.5']:+.1f}", pmed=f"{q['0.5']:,.0f}",
              r25=f"{rq['0.25']:+.1f}", r75=f"{rq['0.75']:+.1f}",
              r05=f"{rq['0.05']:+.1f}", r95=f"{rq['0.95']:+.1f}"))

tab1, tab2, tab3, tab4, tab5 = st.tabs(t("tabs"))

# ----------------------------------------------------------------- fan chart
with tab1:
    fp = fc["fan_path"]; mo = [0] + fp["months"]
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
    st.plotly_chart(fig, use_container_width=True)
    st.info(t("fan_info", w0=cal["window"][0], w1=cal["window"][1], n=cal["n"],
              c80=f"{cal['cover80']:.0%}", c90=f"{cal['cover90']:.0%}"))

# ---------------------------------------------------------------- heatmap
with tab2:
    df = pd.DataFrame(fc["indicators"])
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
    st.caption(t("heat_caption"))

# ---------------------------------------------------------------- calibration
with tab3:
    st.subheader(t("cal_h"))
    st.write(t("cal_intro"))
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(t("cal_m50"), f"{cal.get('cover50', float('nan')):.0%}", t("cal_t50"))
    cc2.metric(t("cal_m80"), f"{cal['cover80']:.0%}", t("cal_t80"))
    cc3.metric(t("cal_m90"), f"{cal['cover90']:.0%}", t("cal_t90"))
    pit = cal.get("pit_hist", [])
    if pit:
        n = sum(pit); k = len(pit); ideal = n / k
        figp = go.Figure()
        figp.add_trace(go.Bar(x=[f"{i*10}–{(i+1)*10}%" for i in range(k)], y=pit, marker_color="#1f77b4",
                              name=t("pit_observed")))
        figp.add_hline(y=ideal, line_dash="dash", line_color="grey", annotation_text=t("pit_ideal"))
        figp.update_layout(height=360, xaxis_title=t("pit_x"), yaxis_title=t("pit_y"),
                           margin=dict(t=20), showlegend=False)
        st.plotly_chart(figp, use_container_width=True)
        st.caption(t("pit_caption", ideal=f"{ideal:.0f}", ks=cal["pit_ks"]))

# ---------------------------------------------------------------- answer-key
with tab4:
    st.subheader(t("ak_h"))
    h = pd.DataFrame(hist["records"])
    st.write(t("ak_intro", n=hist["n"], hit=f"{hist['hit_rate_90']:.0%}"))
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
    cs = pd.DataFrame(fc.get("case_studies", []))
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
        st.caption(t("cs_caption"))
    cols_order = ["origin", "spot", "fc_median", "fc_lo90", "fc_hi90", "realized", "realized_ret_pct", "in90"]
    st.dataframe(h[cols_order].rename(columns=T[L]["tbl"]).iloc[::-1],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- methodology
with tab5:
    st.subheader(t("mth_h"))
    st.markdown(t("mth_body"))
