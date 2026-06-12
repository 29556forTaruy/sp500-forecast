#!/usr/bin/env python
"""Phase 2 data collection pipeline for the S&P 500 forecasting project.

Outputs (data/processed/):
  shiller_monthly.csv  Shiller ie_data monthly series, 1871-
  fred_monthly.csv     14 FRED series aligned to month-end
  spx_daily.csv        ^GSPC daily OHLCV, 1927-
  annual_pivots.csv    yearly pivot levels from prior-year H/L/C
  master_monthly.csv   all series merged on month-end dates

Raw downloads are cached under data/raw/. Run: uv run python fetch_sp500_data.py
"""

import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) sp500-research/0.1"}

SHILLER_YALE_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_PORTAL_URL = "https://shillerdata.com/"

# Series IDs chosen to cover the indicator catalog in PLAN.md §3
# (rates/policy, macro, credit/risk). All work with the keyless fredgraph.csv endpoint.
FRED_SERIES = {
    "FEDFUNDS": "Federal funds effective rate (monthly, %)",
    "DGS10": "10y Treasury constant-maturity yield (daily, %)",
    "DGS2": "2y Treasury constant-maturity yield (daily, %)",
    "T10Y3M": "10y minus 3m Treasury spread (daily, %)",
    "T10Y2Y": "10y minus 2y Treasury spread (daily, %)",
    "DFII10": "10y TIPS real yield (daily, %)",
    "WALCL": "Fed total assets (weekly, $mn)",
    "M2SL": "M2 money stock (monthly, $bn)",
    "UNRATE": "Unemployment rate (monthly, %)",
    "INDPRO": "Industrial production index (monthly)",
    "UMCSENT": "U.Michigan consumer sentiment (monthly)",
    "BAMLH0A0HYM2": "ICE BofA US high-yield OAS (daily, %)",
    "VIXCLS": "CBOE VIX close (daily)",
    "NFCI": "Chicago Fed national financial conditions index (weekly)",
}


def month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def get_with_retry(url: str, tries: int = 3, timeout: int = 90):
    """GET with a real-browser TLS fingerprint (curl_cffi). FRED's WAF silently
    times out / 504s python-requests clients once they request large series."""
    from curl_cffi import requests as browser_requests

    for attempt in range(1, tries + 1):
        try:
            r = browser_requests.get(url, impersonate="chrome", timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == tries:
                raise
            wait = min(20 * 2 ** (attempt - 1), 120)
            print(f"  retry {attempt}/{tries - 1} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- Shiller

def shiller_last_month(path: Path) -> pd.Timestamp | None:
    try:
        df = parse_shiller(path)
        return df["date"].max()
    except Exception:
        return None


def cached_today(path: Path) -> bool:
    if not path.exists():
        return False
    mtime_local = pd.Timestamp.fromtimestamp(path.stat().st_mtime)  # local clock
    return mtime_local.date() == pd.Timestamp.now().date()


def download_shiller() -> Path:
    """Fetch ie_data.xls. Yale first; if stale (>120 days behind today) or dead,
    follow the download link on shillerdata.com (Yale copy stopped updating in 2023)."""
    dest = RAW / "ie_data.xls"
    if cached_today(dest):
        print("  using today's cached copy")
        return dest
    candidates: list[bytes] = []
    try:
        r = requests.get(SHILLER_YALE_URL, headers=UA, timeout=60)
        if r.status_code == 200 and len(r.content) > 100_000:
            candidates.append(r.content)
    except requests.RequestException as e:
        print(f"  yale download failed: {e}")

    def vintage(content: bytes) -> pd.Timestamp:
        tmp = RAW / "_probe.xls"
        tmp.write_bytes(content)
        last = shiller_last_month(tmp)
        tmp.unlink(missing_ok=True)
        return last if last is not None else pd.Timestamp("1900-01-01")

    fresh_enough = pd.Timestamp.today() - pd.Timedelta(days=120)
    if candidates and vintage(candidates[0]) >= fresh_enough:
        dest.write_bytes(candidates[0])
        print("  using Yale copy")
        return dest

    time.sleep(0.6)
    page = requests.get(SHILLER_PORTAL_URL, headers=UA, timeout=60)
    page.raise_for_status()
    links = re.findall(r'href="(//img1\.wsimg\.com/[^"]*ie_data\.xls[^"]*)"', page.text)
    if not links:
        raise RuntimeError("could not locate ie_data.xls link on shillerdata.com")
    time.sleep(0.6)
    r = requests.get("https:" + links[0], headers=UA, timeout=120)
    r.raise_for_status()
    candidates.append(r.content)

    best = max(candidates, key=vintage)
    dest.write_bytes(best)
    print(f"  using {'shillerdata.com' if best is candidates[-1] else 'Yale'} copy")
    return dest


def parse_shiller(path: Path) -> pd.DataFrame:
    """Parse the 'Data' sheet. Header is on the row whose first cell is 'Date';
    dates are floats like 1871.01 where .1 means October (.10 collapses to .1)."""
    raw = pd.read_excel(path, sheet_name="Data", header=None)
    header_rows = raw.index[raw[0].astype(str).str.strip() == "Date"]
    if len(header_rows) == 0:
        raise RuntimeError("Shiller sheet: 'Date' header row not found")
    body = raw.iloc[header_rows[0] + 1 :].copy()

    # positional layout (validated against 2023 and 2026 vintages):
    # 0 Date, 1 P, 2 D, 3 E, 4 CPI, 5 date fraction, 6 GS10,
    # 7 real P, 8 real D, 9 real TR price, 10 real E, 11 real TR scaled E, 12 CAPE
    cols = {0: "datefloat", 1: "P", 2: "D", 3: "E", 4: "CPI", 6: "GS10",
            7: "real_P", 8: "real_D", 10: "real_E", 12: "CAPE"}
    df = body[list(cols)].rename(columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["datefloat", "P"])

    years = df["datefloat"].astype(int)
    months = ((df["datefloat"] - years) * 100).round().astype(int)
    ok = months.between(1, 12)
    if not ok.all():
        raise RuntimeError(f"Shiller sheet: {(~ok).sum()} rows with unparseable month")
    df["date"] = [month_end(y, m) for y, m in zip(years, months)]

    # E10 (10y avg real earnings, latest dollars) recovered exactly via CAPE's definition
    df["E10"] = df["real_P"] / df["CAPE"]

    df = df[["date", "P", "D", "E", "CPI", "GS10",
             "real_P", "real_D", "real_E", "E10", "CAPE"]].reset_index(drop=True)

    first = df.iloc[0]
    assert first["date"] == pd.Timestamp("1871-01-31"), "Shiller data must start 1871-01"
    assert abs(first["P"] - 4.44) < 0.01, "unexpected P for 1871-01"
    cape2000 = df.loc[df["date"] == "2000-01-31", "CAPE"].iloc[0]
    assert 43.0 < cape2000 < 45.0, f"CAPE Jan-2000 sanity failed: {cape2000}"
    return df


# ------------------------------------------------------------------- FRED

# Daily series choke FRED's CSV backend when requested whole (504/timeouts that
# scale with row count), so they are fetched in 4-year chunks from their start year.
FRED_DAILY_START = {
    "DGS10": 1962, "DGS2": 1976, "T10Y3M": 1982, "T10Y2Y": 1976,
    "DFII10": 2003, "BAMLH0A0HYM2": 1996, "VIXCLS": 1990,
}


def fetch_fred_series(series: str) -> bytes:
    base = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    if series not in FRED_DAILY_START:
        return get_with_retry(base, tries=3, timeout=60).content
    end_year = pd.Timestamp.today().year
    lines: list[str] = []
    for y0 in range(FRED_DAILY_START[series], end_year + 1, 4):
        y1 = min(y0 + 3, end_year)
        url = f"{base}&cosd={y0}-01-01&coed={y1}-12-31"
        chunk = get_with_retry(url, tries=3, timeout=90).text.strip().splitlines()
        if len(chunk) > 1 and chunk[1].split(",")[0] > f"{y1}-12-31":
            # server ignored cosd/coed (licensed series, e.g. ICE BofA, are clamped
            # to a trailing window on the keyless endpoint) — keep that window only
            print(f"    {series}: fredgraph ignores cosd/coed; only "
                  f"{chunk[1].split(',')[0]}..{chunk[-1].split(',')[0]} available", flush=True)
            return ("\n".join(chunk) + "\n").encode()
        if not lines:
            lines.append(chunk[0])  # header once
        lines.extend(chunk[1:])
        print(f"    {series} {y0}-{y1}: {len(chunk) - 1} rows", flush=True)
        time.sleep(1.5)
    return ("\n".join(lines) + "\n").encode()


def fetch_fred() -> pd.DataFrame:
    """Keyless fredgraph.csv endpoint. Missing values are '.'; the date column
    name varies by version so both columns are taken by position."""
    monthly = {}
    failed = []
    for series in FRED_SERIES:
        cache = RAW / f"fred_{series}.csv"
        if not cached_today(cache):
            time.sleep(2.0)
            try:
                content = fetch_fred_series(series)
            except Exception as e:
                print(f"  {series}: FAILED ({e}); continuing — rerun later to fill", flush=True)
                failed.append(series)
                continue
            cache.write_bytes(content)
        df = pd.read_csv(cache, na_values=".")
        s = pd.Series(
            pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
            index=pd.to_datetime(df.iloc[:, 0]),
            name=series,
        )
        monthly[series] = s.resample("ME").last()  # last valid observation in month
        print(f"  {series}: {s.index.min().date()} .. {s.index.max().date()} ({len(s)} obs)", flush=True)
    if failed:
        raise RuntimeError(
            f"FRED series failed: {', '.join(failed)} — fetched ones are cached in data/raw; rerun to fill the rest")
    out = pd.DataFrame(monthly)
    out.index.name = "date"
    return out.reset_index()


# -------------------------------------------------------------- yfinance

def fetch_spx() -> pd.DataFrame:
    import yfinance as yf

    df = yf.download("^GSPC", period="max", auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        df = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance: (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = (df[["Open", "High", "Low", "Close", "Volume"]]
          .rename(columns=str.lower)
          .reset_index()
          .rename(columns={"Date": "date", "index": "date"}))
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    assert df["date"].iloc[0].year <= 1928, "^GSPC history should start by 1928"
    return df


# ---------------------------------------------------------------- pivots

def annual_pivots(spx: pd.DataFrame) -> pd.DataFrame:
    """Pivot levels for year Y from year Y-1's high/low/close.
    The row built from the in-progress year is flagged provisional."""
    g = spx.groupby(spx["date"].dt.year)
    yearly = pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "last_date": g["date"].max(),
    })
    rows = []
    for src_year, r in yearly.iterrows():
        H, L, C = r["high"], r["low"], r["close"]
        P = (H + L + C) / 3
        complete = r["last_date"] >= pd.Timestamp(src_year, 12, 28)
        rows.append({
            "year": src_year + 1,
            "prev_high": H, "prev_low": L, "prev_close": C,
            "P": P, "R1": 2 * P - L, "S1": 2 * P - H,
            "R2": P + (H - L), "S2": P - (H - L),
            "provisional": not complete,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- master

def build_master(shiller: pd.DataFrame, fred: pd.DataFrame, spx: pd.DataFrame) -> pd.DataFrame:
    sh = shiller.set_index("date").add_prefix("shiller_")
    fr = fred.set_index("date")
    px = spx.set_index("date")["close"].resample("ME").last().rename("spx_close")
    master = sh.join(fr, how="outer").join(px, how="outer").sort_index()
    master.index.name = "date"
    return master.reset_index()


# ---------------------------------------------------------------- report

def coverage_table(frames: dict[str, pd.DataFrame]) -> str:
    lines = ["| ファイル | 列 | 開始 | 最終 | 行数(非欠損) | 欠損率(自系列期間内) |",
             "|---|---|---|---|---|---|"]
    for fname, df in frames.items():
        for col in df.columns:
            if col in ("date", "year", "provisional"):
                continue
            s = df.set_index(df.columns[0])[col]
            valid = s.dropna()
            if valid.empty:
                lines.append(f"| {fname} | {col} | - | - | 0 | - |")
                continue
            span = s.loc[valid.index.min(): valid.index.max()]
            miss = 1 - len(valid) / len(span)
            lines.append(
                f"| {fname} | {col} | {valid.index.min().date() if hasattr(valid.index.min(), 'date') else valid.index.min()} "
                f"| {valid.index.max().date() if hasattr(valid.index.max(), 'date') else valid.index.max()} "
                f"| {len(valid)} | {miss:.2%} |")
    return "\n".join(lines)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] Shiller ie_data.xls ...")
    shiller = parse_shiller(download_shiller())
    shiller.to_csv(OUT / "shiller_monthly.csv", index=False)
    print(f"  -> shiller_monthly.csv {shiller.shape}, last={shiller['date'].max().date()}")

    print("[2/5] ^GSPC daily via yfinance ...")
    spx_path = OUT / "spx_daily.csv"
    if cached_today(spx_path):
        spx = pd.read_csv(spx_path, parse_dates=["date"])
        print("  using today's cached copy")
    else:
        spx = fetch_spx()
        spx.to_csv(spx_path, index=False)
    print(f"  -> spx_daily.csv {spx.shape}, last={spx['date'].max().date()} close={spx['close'].iloc[-1]:.2f}")

    print("[3/5] FRED (14 series) ...")
    fred = fetch_fred()
    fred.to_csv(OUT / "fred_monthly.csv", index=False)
    print(f"  -> fred_monthly.csv {fred.shape}")

    print("[4/5] annual pivots ...")
    pivots = annual_pivots(spx)
    pivots.to_csv(OUT / "annual_pivots.csv", index=False)
    print(f"  -> annual_pivots.csv {pivots.shape}")

    print("[5/5] master monthly ...")
    master = build_master(shiller, fred, spx)
    master.to_csv(OUT / "master_monthly.csv", index=False)
    print(f"  -> master_monthly.csv {master.shape}")

    table = coverage_table({
        "shiller_monthly": shiller,
        "fred_monthly": fred,
        "spx_daily": spx,
        "master_monthly": master,
    })
    (OUT / "_coverage_table.md").write_text(table)
    print("coverage table written to data/processed/_coverage_table.md")


if __name__ == "__main__":
    sys.exit(main())
