# -*- coding: utf-8 -*-
"""FX conversion + alignment helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

def to_jpy(prices_local: pd.DataFrame, meta_df: pd.DataFrame, usdjpy: pd.Series) -> pd.DataFrame:
    """Convert local-currency prices to JPY using USDJPY.

    Supported currencies: JPY, USD. For USD instruments, multiply by USDJPY.
    meta_df must contain at least columns: ticker, currency.
    """
    if prices_local is None or prices_local.empty:
        return pd.DataFrame(index=getattr(prices_local, "index", None))

    # Normalize USDJPY
    if isinstance(usdjpy, pd.DataFrame):
        usdjpy = usdjpy.iloc[:, 0]
    usdjpy = pd.Series(usdjpy).squeeze()
    usdjpy.index = pd.to_datetime(usdjpy.index)
    try:
        usdjpy.index = usdjpy.index.tz_localize(None)
    except Exception:
        pass
    usdjpy = usdjpy.sort_index()

    # Forward fill USDJPY to price index
    usdjpy_ff = usdjpy.reindex(prices_local.index.union(usdjpy.index)).sort_index().ffill()
    usdjpy_ff = usdjpy_ff.reindex(prices_local.index).ffill()

    meta = meta_df.copy()
    if "currency" not in meta.columns:
        raise ValueError("meta_df must include 'currency' column.")
    meta["currency"] = meta["currency"].astype(str).str.upper()
    meta = meta.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")

    out = pd.DataFrame(index=prices_local.index)
    for tkr in prices_local.columns:
        if tkr not in meta.index:
            # Default to USD if unknown (safer for US tickers); user can override in portfolio metadata
            cur = "USD"
        else:
            cur = meta.loc[tkr, "currency"]
            if isinstance(cur, pd.Series):
                cur = cur.iloc[0]
            cur = str(cur).upper()

        s = prices_local[tkr]
        if cur == "JPY":
            out[tkr] = s
        elif cur == "USD":
            out[tkr] = s * usdjpy_ff
        else:
            raise ValueError(f"Unsupported currency: {cur} ({tkr})")
    return out


def align_common_start(prices: pd.DataFrame, requested_start: str) -> pd.DataFrame:
    """
    - requested_start以降で、全銘柄が揃う最初の日から開始
    - 同日に揃わない場合は ffill で救済（市場休場差など）
    - どれかの列が全欠損なら明示的にエラー
    """
    requested_start = pd.to_datetime(requested_start)
    p = prices.loc[prices.index >= requested_start].copy()

    # Columns with no data in the selected window are common (e.g., delisted, wrong suffix,
    # newly listed, or simply no overlap with the chosen period). Historically this raised
    # and broke the whole app. Instead, we drop those columns and surface a warning in the UI.
    all_nan_cols = [c for c in p.columns if p[c].dropna().empty]
    if all_nan_cols:
        p = p.drop(columns=all_nan_cols, errors="ignore")
        # Attach metadata for the caller (Streamlit UI) to warn the user.
        try:
            p.attrs["dropped_no_data_tickers"] = list(all_nan_cols)
        except Exception:
            pass
        # If everything is gone, do not hard-crash the whole app.
        # Return an empty frame with metadata so the caller can surface a friendly message.
        if p.empty or p.shape[1] == 0:
            empty = pd.DataFrame(index=p.index)
            try:
                empty.attrs["dropped_no_data_tickers"] = list(all_nan_cols)
                empty.attrs["all_no_data"] = True
            except Exception:
                pass
            return empty

    valid = p.dropna(how="any")
    if not valid.empty:
        common_start = valid.index.min()
        return p.loc[p.index >= common_start].dropna(how="any")

    p_ffill = p.sort_index().ffill()
    valid2 = p_ffill.dropna(how="any")
    if valid2.empty:
        # Same philosophy: avoid raising; let UI decide how to message.
        empty = pd.DataFrame(index=p_ffill.index)
        try:
            empty.attrs["no_overlap"] = True
        except Exception:
            pass
        return empty
    common_start = valid2.index.min()
    return p_ffill.loc[p_ffill.index >= common_start].dropna(how="any")


# =========================
# Portfolio helpers
# =========================
