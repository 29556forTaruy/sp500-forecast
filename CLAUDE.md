# S&P 500 予測モデル — データ基盤リポジトリ

## プロジェクト概要

S&P 500 の今後1年の値動きを確率分布(分位点+ファンチャート)として予測するモデルと、毎日自動更新されるアプリを段階的に開発する。全体計画・モデル仕様は `PLAN.md` を正とする。フェーズ2(データ収集)完了。フェーズ3–5でモデリングのはしご(Level 0–5)を一通り実証(PLAN.md §11–§13):Level 0 構造アンカー=1年点予測の最善(`phase3_level0.py`、§11)/Level 2 マクロは点予測に signal を足さず棄却(`phase4_level2.py`、§12)/**Level 5 = GARCH ボラ + FHS 形状で較正済みの確率的ファンチャート=最終product**(`phase5_fanchart.py`、§13)。data/processed の生の収集CSV は「生の値の月末整列」まで(発表ラグ調整は特徴量・バックテスト側で実施)。

## ディレクトリ構成

```
.
├── PLAN.md               # 開発計画(モデル仕様 §5.1、§11=Level 0 実証結果、付録A=設計判断の根拠)
├── fetch_sp500_data.py   # フェーズ2 データ収集パイプライン(uv run python fetch_sp500_data.py)
├── phase3_level0.py      # フェーズ3 Level 0 特徴量+ウォークフォワード・バックテスト(uv run python phase3_level0.py)
├── phase4_level2.py      # フェーズ4 Level 2 ElasticNet残差モード検証(uv run python phase4_level2.py)
├── phase5_fanchart.py    # フェーズ5 Level 5 確率的ファンチャート・較正検証(uv run python phase5_fanchart.py)
├── DATA_REPORT.md        # 収集結果レポート(各系列のカバレッジ、サニティチェック)
├── pyproject.toml        # uv プロジェクト(Python 3.11+; pandas, requests, yfinance, xlrd, numpy, statsmodels, matplotlib, curl-cffi)
└── data/
    ├── raw/              # ダウンロード生データのキャッシュ(ie_data.xls, fred_*.csv)
    └── processed/        # 出力CSV(下記スキーマ)+ level0_* (フェーズ3成果物)
```

## 出力CSVのスキーマ (data/processed/)

依頼書記載の「たたき台スクリプト」はリポジトリに存在しなかったため、スキーマは 2026-06-13 に本実装で定義した。**以後このスキーマを正とし、変更時は理由を記録すること。**

### shiller_monthly.csv (1871-01〜, 月次)
`date, P, D, E, CPI, GS10, real_P, real_D, real_E, E10, CAPE`
- `date`: 月末日付 (YYYY-MM-DD)。`P/D/E`: 名目の価格/配当/EPS。`real_*`: 最新CPI基準の実質値
- `E10`: 実質10年平均EPS(最新ドル表示)。`real_P / CAPE` で復元した値で、CAPE の定義と厳密に整合
- `CAPE`: Shiller PE(1881-01から。E10が貯まるまでの最初の10年はNaN)

### fred_monthly.csv (系列により開始が異なる, 月次)
`date` + 14系列(列名 = FRED系列ID):
`FEDFUNDS, DGS10, DGS2, T10Y3M, T10Y2Y, DFII10, WALCL, M2SL, UNRATE, INDPRO, UMCSENT, BAMLH0A0HYM2, VIXCLS, NFCI`
- 日次・週次系列は**月内最後の有効観測値**で月末に整列。月次系列はその月の値を月末日付に付け替え
- 系列の選定根拠: PLAN.md §3 の指標カタログ(金利・金融政策/景気マクロ/クレジット・リスク)

### spx_daily.csv (1927-12-30〜, 日次)
`date, open, high, low, close, volume`
- yfinance `^GSPC`、無調整 OHLCV(`auto_adjust=False`)

### annual_pivots.csv (年次)
`year, prev_high, prev_low, prev_close, P, R1, S1, R2, S2, provisional`
- `year` の行は **前年(year−1)** の高値H・安値L・終値Cから算出: P=(H+L+C)/3, R1=2P−L, S1=2P−H, R2=P+(H−L), S2=P−(H−L)
- `provisional=True`: 進行中の年のデータから作った「来年用」の行(年が完了したら再計算で確定)
- 注意: 1928年の行は1927年の2営業日(^GSPCの収録開始が1927-12-30)のみから計算されており実用不可

### master_monthly.csv (1871-01〜, 月次・月末ベース外部結合)
`date` + `shiller_P, shiller_D, shiller_E, shiller_CPI, shiller_GS10, shiller_real_P, shiller_real_D, shiller_real_E, shiller_E10, shiller_CAPE` + FRED 14系列(IDのまま) + `spx_close`(^GSPC 月内最終終値)

## 既知のデータの罠(必読)

1. **Shiller の日付列は float**: `1871.01`形式で、`.1` は「10月」(`.10` が float 化で `.1` になる)。`月 = round((値 − 年) × 100)` で復元する
2. **Yale の ie_data.xls は2023年9月で更新停止**。配布元は https://shillerdata.com/ に移転(実体は `img1.wsimg.com/blobby/...` のCDNリンクで、URLは更新ごとに変わる → ページをスクレイプしてリンクを辿る)。パイプラインは Yale 取得→鮮度チェック→古ければ shillerdata.com の順
3. **Shiller の P は日次終値の月中平均**(1926年以降。それ以前はCowles指数由来)。**最新月は単日の値**のことがある(例: 2026-06 行は6月初の値)。yfinance の月末終値と混ぜるときは系統差に注意
4. **Shiller の E は四半期報告の線形補間+公表ラグ**(確報まで1〜2四半期)。直近数ヶ月の E・CAPE は推定値を含む。バックテストでは E 由来列を3ヶ月ラグさせた変種を正とする(PLAN.md §6)
5. **`P/CAPE`(=E10)は「CPI実質化した10年平均EPSの t 時点ドル表示」**であり、名目EPSの単純10年平均ではない(乖離 中央値~11%)
6. **FRED の欠損値は `.`(ピリオド)**。先頭列の列名はバージョンで変わるため位置で扱う
7. **FRED fredgraph.csv はキー不要だが脆い**: ①python-requests のTLSフィンガープリントがWAFに弾かれることがある → curl_cffi(`impersonate="chrome"`)を使用 ②日次系列は全期間一括だとサーバ側でタイムアウト(504)する → 4年チャンク(`cosd`/`coed`)で分割取得 ③旧 `downloaddata` エンドポイントは廃止済み(壊れたリダイレクトを返す)。当日キャッシュ(`data/raw/fred_*.csv`)+バックオフ付きリトライで対応。連続アクセスは最低0.8秒間隔
8. **ライセンス系列のクランプ**: ICE BofA 系列(BAMLH0A0HYM2 等)は fredgraph.csv が `cosd`/`coed` を無視して**直近約3年分しか返さない**。スクリプトはクランプを検出してその窓だけ保存する。全履歴(1996年〜)が必要になったら無料の FRED API キー登録が必要(フェーズ3で判断。長期クレジットスプレッドの代替候補: Moody's BAA−AAA)
9. **yfinance は単一ティッカーでも MultiIndex 列を返すことがある** → `columns.get_level_values(0)` でフラット化。**1962年以前の ^GSPC は全日 High=Low=Close**(日中レンジなし)で、その後も**1983年6月まで散発的にフラットな日が38日**ある(1967年・1971年に集中)。1950年以前は volume=0。また **2023-05-24 の volume が 0**(Yahoo 側の単発グリッチ、価格は正常)
10. **FRED 実データの既知の穴**: UNRATE 2025-10 が欠損(政府閉鎖で家計調査が実施されず — ソース自体に存在しない)。UMCSENT は1978年以前が四半期調査のため月次では~24%欠損
11. **重複ウィンドウ・Stambaughバイアス等の統計的罠**は PLAN.md §4・付録A 参照(モデリング時)

## 開発ルール

- 環境: `uv sync` で再現。実行: `uv run python fetch_sp500_data.py`(当日キャッシュがあれば再ダウンロードしない)
- 外部サービスへの過剰アクセス禁止(リトライはバックオフ付き、FRED は0.8秒間隔)
- APIキーが必要な手段は現段階では使わない(fredgraph.csv で足りる)
- processed CSV のスキーマ変更時は DATA_REPORT.md と本ファイルの両方を更新し、理由を明記
