# -*- coding: utf-8 -*-
"""Persistence for portfolios.json.

This module intentionally keeps I/O and light normalization separate from UI.
"""

from __future__ import annotations

import json
import os

import pandas as pd


def _portfolios_default() -> dict:
    return {
        "version": 1,
        "updated_at": "",
        "portfolios": {},
    }


def load_portfolios(path: str) -> dict:
    """Load portfolios.json, returning a normalized dict even if missing/corrupt."""
    try:
        if not os.path.exists(path):
            return _portfolios_default()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return _portfolios_default()

        if "portfolios" not in data or not isinstance(data.get("portfolios"), dict):
            data["portfolios"] = {}

        data.setdefault("version", 1)
        data.setdefault("updated_at", "")
        return data
    except Exception:
        return _portfolios_default()


def save_portfolios(path: str, data: dict) -> None:
    """Save portfolios.json (best-effort atomic replace)."""
    data = dict(data)
    data["updated_at"] = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, path)


def portfolio_to_df_items(pf_items: list[dict]) -> pd.DataFrame:
    """Convert a portfolio's items list into a tidy DataFrame."""
    if not pf_items:
        return pd.DataFrame(columns=["display_name", "ticker", "asset_type", "currency"])

    df = pd.DataFrame(pf_items)
    for col in ["display_name", "ticker", "asset_type", "currency"]:
        if col not in df.columns:
            df[col] = ""

    df = df[["display_name", "ticker", "asset_type", "currency"]].copy()
    df["display_name"] = df["display_name"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["asset_type"] = df["asset_type"].astype(str).str.strip()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()

    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="first")
    return df
