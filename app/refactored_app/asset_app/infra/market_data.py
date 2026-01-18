# -*- coding: utf-8 -*-
"""Market data fetchers (prices, FX, market caps)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import json
from pathlib import Path

from asset_app.config.paths import default_us_mcap_cache_path

from ..config.constants import REP_MARKET_UNIVERSE

def guess_asset_type_from_quote_type(quote_type: str) -> str:
    """Map Yahoo Finance quoteType to our internal asset_type.

    We keep the mapping intentionally simple and safe; unknown types fall back to 'equity'.
    """
    qt = (quote_type or "").strip().upper()
    if qt in {"ETF"}:
        return "etf"
    if qt in {"MUTUALFUND", "MUTUAL_FUND"}:
        return "fund"
    if qt in {"CRYPTOCURRENCY", "CRYPTO"}:
        return "crypto"
    if qt in {"INDEX"}:
        return "index"
    if qt in {"EQUITY", "STOCK"}:
        return "equity"
    # Defaults
    return "equity"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_adjclose(
    tickers: list[str] | tuple[str, ...],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch adjusted close prices via yfinance.

    Returns a DataFrame indexed by datetime, columns=tickers, values=Adj Close (or Close if Adj Close unavailable).
    This function is cached; make sure to include all inputs in the signature.
    """
    if isinstance(tickers, (list, tuple)):
        tks = [str(t).strip() for t in tickers if str(t).strip()]
    else:
        tks = [str(tickers).strip()]
    if not tks:
        return pd.DataFrame()

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    try:
        df = yf.download(
            tks,
            start=start_ts,
            end=end_ts,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance download failed: {e}")

    if df is None or df.empty:
        return pd.DataFrame()

    # yf.download returns:
    # - MultiIndex columns when multiple tickers
    # - SingleIndex columns when single ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Adj Close; fallback to Close
        if ("Adj Close" in df.columns.get_level_values(0)):
            px = df["Adj Close"].copy()
        elif ("Close" in df.columns.get_level_values(0)):
            px = df["Close"].copy()
        else:
            # take last level heuristic
            px = df.xs(df.columns.levels[0][0], axis=1, level=0).copy()
        # Ensure all tickers exist as columns
        px = px.reindex(columns=tks)
    else:
        # Single ticker
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].rename(columns={"Adj Close": tks[0]}).copy()
        elif "Close" in df.columns:
            px = df[["Close"]].rename(columns={"Close": tks[0]}).copy()
        else:
            px = df[[df.columns[0]]].rename(columns={df.columns[0]: tks[0]}).copy()

    # Normalize index name/timezone and sort
    px.index = pd.to_datetime(px.index)
    try:
        px.index = px.index.tz_localize(None)
    except Exception:
        pass
    px = px.sort_index()
    return px


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usdjpy(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """Fetch USDJPY (JPY per 1 USD) time series.

    Uses Yahoo Finance ticker "JPY=X". Returns a Series named 'USDJPY'.
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    fx = yf.download(
        tickers="JPY=X",
        start=start_ts,
        end=end_ts,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if fx is None or fx.empty:
        raise RuntimeError("Failed to fetch USDJPY (JPY=X) from Yahoo Finance.")

    if isinstance(fx.columns, pd.MultiIndex):
        if ("Adj Close" in fx.columns.get_level_values(0)):
            s = fx["Adj Close"].copy()
        elif ("Close" in fx.columns.get_level_values(0)):
            s = fx["Close"].copy()
        else:
            s = fx.xs(fx.columns.levels[0][0], axis=1, level=0).copy()
        # single ticker, make it a Series
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
    else:
        if "Adj Close" in fx.columns:
            s = fx["Adj Close"].copy()
        elif "Close" in fx.columns:
            s = fx["Close"].copy()
        else:
            s = fx.iloc[:, 0].copy()

    s = pd.Series(s).squeeze()
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_localize(None)
    except Exception:
        pass
    s = s.sort_index()
    s.name = "USDJPY"
    return s


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_market_caps(tickers: list[str] | tuple[str, ...]) -> dict:
    """Fetch market-cap (or AUM-like) proxies for tickers.

    This is used only to construct *relative* weights for Black-Litterman implied prior.
    Yahoo Finance coverage is imperfect (especially for crypto/index), so this function is
    intentionally best-effort and returns NaN when unavailable.

    Returns
    -------
    dict
        {ticker: market_cap_float_or_nan}
    """
    tks = [str(t).strip() for t in (list(tickers) if isinstance(tickers, (list, tuple)) else [tickers])]
    out: dict[str, float] = {}
    for t in tks:
        if not t:
            continue
        mc = np.nan
        try:
            tk = yf.Ticker(t)
            # Prefer fast_info if present (lighter)
            fi = getattr(tk, "fast_info", None)
            if isinstance(fi, dict):
                mc = fi.get("market_cap")
            if mc is None or (isinstance(mc, float) and np.isnan(mc)):
                # Fallback to info (heavier)
                info = getattr(tk, "info", None) or {}
                if isinstance(info, dict):
                    mc = info.get("marketCap")
        except Exception:
            mc = np.nan

        try:
            out[t] = float(mc) if mc is not None else np.nan
        except Exception:
            out[t] = np.nan

    # Persist to local cache (best-effort). This helps improve local search ranking
    # without triggering extra network calls.
    try:
        cache_path: Path = default_us_mcap_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, float] = {}
        if cache_path.exists():
            try:
                existing_raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(existing_raw, dict):
                    for k, v in existing_raw.items():
                        try:
                            existing[str(k).strip().upper()] = float(v)
                        except Exception:
                            continue
            except Exception:
                existing = {}

        for k, v in out.items():
            kk = str(k).strip().upper()
            if not kk:
                continue
            try:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    existing[kk] = float(v)
            except Exception:
                continue

        cache_path.write_text(json.dumps(existing, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_rep_market_prices_jpy(start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch representative market universe prices and convert to JPY.

    Uses REP_MARKET_UNIVERSE (hard-coded asset-class proxies) and returns a DataFrame
    with columns=tickers in JPY terms. USD assets are converted using USDJPY (JPY=X).

    Notes
    -----
    - This is intentionally independent from instruments.csv. instruments.csv is kept for
      internal benchmark / other purposes, while this representative universe is a
      stable, small proxy set for BL market model.
    """
    rep = REP_MARKET_UNIVERSE
    rep_tickers = [r["ticker"] for r in rep]
    px = fetch_adjclose(rep_tickers, start=start, end=end, interval=interval)
    if px is None or px.empty:
        return pd.DataFrame()

    # FX for USD->JPY conversion
    need_fx = any(str(r.get("currency", "")).upper() == "USD" for r in rep)
    fx = None
    if need_fx:
        fx = fetch_usdjpy(start=start, end=end)
        fx = fx.reindex(px.index).ffill().bfill()

    out = pd.DataFrame(index=px.index)
    for r in rep:
        t = r["ticker"]
        cur = str(r.get("currency", "JPY")).upper()
        s = px[t].copy() if t in px.columns else None
        if s is None:
            continue
        if cur == "USD":
            if fx is None or fx.empty:
                continue
            s = s.astype(float) * fx.astype(float)
        out[t] = s.astype(float)

    out = out.dropna(how="all")
    return out

