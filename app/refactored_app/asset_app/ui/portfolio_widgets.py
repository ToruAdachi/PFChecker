# -*- coding: utf-8 -*-
"""Sidebar portfolio widgets helpers.

These helpers intentionally remain Streamlit-dependent.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from asset_app.infra.portfolio_store import portfolio_to_df_items


def compute_weights_from_weight_widgets(pf_items: list[dict]) -> dict:
    """Return normalized weights {ticker: weight} based on current UI inputs.

    Reads st.session_state['w_<display_name>'].
    """
    df = portfolio_to_df_items(pf_items)
    if df.empty:
        return {}

    tickers = df["ticker"].astype(str).tolist()
    dns = df["display_name"].astype(str).tolist()

    raw = []
    for dn in dns:
        v = st.session_state.get(f"w_{dn}")
        try:
            raw.append(float(v) if v is not None else 0.0)
        except Exception:
            raw.append(0.0)

    w = np.array(raw, dtype=float)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        eq = 1.0 / len(tickers)
        return {t: float(eq) for t in tickers}

    w = w / w.sum()
    return {t: float(w[i]) for i, t in enumerate(tickers)}


def compute_weights_from_widgets(items: list[dict]) -> dict:
    """Alias of compute_weights_from_weight_widgets for backward compatibility."""
    return compute_weights_from_weight_widgets(items)


def get_current_pf_state() -> tuple[list[dict], dict]:
    items = st.session_state.get("portfolio_items", [])
    weights_by_ticker = st.session_state.get("portfolio_weights", {})
    if not isinstance(items, list):
        items = []
    if not isinstance(weights_by_ticker, dict):
        weights_by_ticker = {}
    return items, weights_by_ticker


def set_current_pf_state(items: list[dict], weights_by_ticker: dict) -> None:
    st.session_state["portfolio_items"] = items
    st.session_state["portfolio_weights"] = weights_by_ticker


def apply_weights_to_weight_widgets(items: list[dict], weights_by_ticker: dict) -> None:
    """Apply ticker-based weights to UI widgets (w_<display_name>) safely."""
    df = portfolio_to_df_items(items)
    if df.empty:
        return

    for _, row in df.iterrows():
        dn = str(row.get("display_name", "")).strip()
        t = str(row.get("ticker", "")).strip()
        if not dn or not t:
            continue
        v = weights_by_ticker.get(t)
        try:
            st.session_state[f"w_{dn}"] = float(v) if v is not None else 0.0
        except Exception:
            st.session_state[f"w_{dn}"] = 0.0
