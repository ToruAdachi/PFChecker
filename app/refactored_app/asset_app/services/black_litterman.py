# -*- coding: utf-8 -*-
"""Black-Litterman helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pypfopt import black_litterman
    _HAVE_PYPORTFOLIOOPT = True
except Exception:
    black_litterman = None
    _HAVE_PYPORTFOLIOOPT = False

def read_bl_settings():
    """
    Read Black-Litterman related settings from Streamlit session_state.

    Returns:
        alpha (float): blend factor (1.0 = BL only, 0.0 = historical mean only)
        tau (float): view confidence scaling (used when views are supplied)
        abs_views (dict[str, float]): absolute views in annual simple return for tickers
        rel_views (list[dict]): relative views: [{a,b,strength,confidence}, ...]
        rel_mode (str): relative Q mode label from UI
        rel_q_base (float): scale multiplier for relative Q
        market_model (str): market model label from UI
        delta (float): risk aversion parameter used for implied prior
    """
    alpha = float(st.session_state.get("_bl_alpha", st.session_state.get("bl_alpha", 1.0)))
    tau = float(st.session_state.get("_bl_tau", st.session_state.get("bl_tau", 0.05)))

    abs_views = st.session_state.get("_bl_views", {}) or {}
    if not isinstance(abs_views, dict):
        try:
            abs_views = dict(abs_views)
        except Exception:
            abs_views = {}
    abs_views = {str(k): float(v) for k, v in abs_views.items() if k is not None}


    abs_view_confs = st.session_state.get("_bl_views_conf", {}) or {}
    if not isinstance(abs_view_confs, dict):
        try:
            abs_view_confs = dict(abs_view_confs)
        except Exception:
            abs_view_confs = {}
    clean_abs_confs = {}
    for k, v in abs_view_confs.items():
        try:
            clean_abs_confs[str(k)] = float(np.clip(float(v), 0.0, 1.0))
        except Exception:
            continue
    rel_views = st.session_state.get("_bl_rel_views", []) or []
    if not isinstance(rel_views, list):
        rel_views = []
    # sanitize
    clean_rel = []
    for rv in rel_views:
        try:
            a = str(rv.get("a", ""))
            b = str(rv.get("b", ""))
            strength = str(rv.get("strength", ">"))
            conf = float(rv.get("confidence", 0.6))
            conf = float(np.clip(conf, 0.0, 1.0))
            if a and b and a != b:
                clean_rel.append({"a": a, "b": b, "strength": strength, "confidence": conf})
        except Exception:
            continue

    rel_mode = str(st.session_state.get("_bl_rel_mode", "固定（>→0.5%/年、>>→2%/年、>>>→5%/年）"))
    rel_q_base = float(st.session_state.get("_bl_rel_q_base", 1.0))

    market_model = str(st.session_state.get("_bl_market_model", "PF時価総額/AUM（簡易）"))
    delta = float(st.session_state.get("_bl_delta", 2.5))

    # sanitize main knobs
    tau = float(np.clip(tau, 1e-6, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rel_q_base = float(np.clip(rel_q_base, 0.1, 5.0))

    return alpha, tau, abs_views, clean_abs_confs, clean_rel, rel_mode, rel_q_base, market_model, delta

def _normalize_weights(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights sum to 0 after clipping.")
    return w / s


# ---- Black-Litterman helpers (Views + blend) ----
def _get_bl_config_from_session():
    """Read BL settings from Streamlit session_state with safe defaults."""
    try:
        views = st.session_state.get("_bl_views", {}) or {}
        tau = float(st.session_state.get("_bl_tau", 0.05))
        alpha = float(st.session_state.get("_bl_alpha", 1.0))  # blend: 1.0=BL only
        market_model = str(st.session_state.get("_bl_market_model", "PF時価総額/AUM（簡易）"))
        delta = float(st.session_state.get("_bl_delta", 2.5))
    except Exception:
        views, tau, alpha = {}, 0.05, 1.0
    # sanitize
    tau = float(np.clip(tau, 1e-6, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    # views: {ticker: annual_return}
    clean_views = {}
    for k, v in views.items():
        try:
            clean_views[str(k)] = float(v)
        except Exception:
            continue
    return clean_views, tau, alpha


def _bl_posterior_returns_from_absolute_views(
    S: pd.DataFrame,
    pi: pd.Series,
    views_abs: dict,
    tau: float,
) -> pd.Series:
    """Compute BL posterior returns given prior pi and absolute views (ticker -> annual return)."""
    if not views_abs:
        return pi

    assets = list(pi.index)
    rows = []
    q = []
    for tkr, ret in views_abs.items():
        if tkr not in assets:
            continue
        p_row = np.zeros(len(assets), dtype=float)
        p_row[assets.index(tkr)] = 1.0
        rows.append(p_row)
        q.append(float(ret))

    if len(rows) == 0:
        return pi

    P = np.vstack(rows)
    Q = np.array(q, dtype=float).reshape(-1, 1)

    # Use explicit pi so market_caps isn't required and Q is always defined.
    bl = black_litterman.BlackLittermanModel(
        S,
        pi=pi,
        P=P,
        Q=Q,
        omega="default",
        tau=tau,
    )
    mu_post = bl.bl_returns()
    # ensure Series aligned
    if isinstance(mu_post, pd.DataFrame):
        mu_post = mu_post.iloc[:, 0]
    return pd.Series(mu_post, index=assets)


def compute_bl_posterior_returns(
    pi: pd.Series,
    S: pd.DataFrame,
    tau: float,
    prices_train: pd.DataFrame | None = None,
    trading_days_per_year: float | None = None,
    abs_views: dict[str, float] | None = None,
    abs_view_confs: dict[str, float] | None = None,
    rel_views: list[dict] | None = None,
    rel_mode: str = "固定（>→0.5%/年、>>→2%/年、>>>→5%/年）",
    rel_q_base: float = 1.0,
) -> pd.Series:
    """Compute BL posterior returns from a prior (pi) and optional views (absolute + relative).

    - If no views are provided, returns pi (prior) without requiring Q.
    - Absolute views: {ticker: annual_expected_return}
    - Relative views: [{"a":tickerA, "b":tickerB, "strength": ">"|">>"|">>>", "confidence":0..1}, ...]

    rel_mode:
      - 固定... : Q is mapped from strength to fixed annual diff (0.5%, 2%, 5%) and scaled by rel_q_base.
      - 自動... : Q is proportional to annualized vol of (A-B) over the training window and scaled by strength & rel_q_base.
    """
    if abs_views is None:
        abs_views = {}
    if rel_views is None:
        rel_views = []

    # Fast path: no views
    if (not abs_views) and (not rel_views):
        return pi

    assets = list(pi.index)
    n = len(assets)
    idx = {a: i for i, a in enumerate(assets)}

    P_rows = []
    Q_vals = []
    confs = []

    # ---- absolute views ----
    for tkr, ret in (abs_views or {}).items():
        if tkr not in idx:
            continue
        row = np.zeros(n, dtype=float)
        row[idx[tkr]] = 1.0
        P_rows.append(row)
        Q_vals.append(float(ret))
        confs.append(float((abs_view_confs or {}).get(tkr, 0.60)))  # default=0.60（中）

    # ---- relative views ----
    fixed_map = {">": 0.005, ">>": 0.02, ">>>": 0.05}  # annual simple diff
    auto_k = {">": 0.25, ">>": 0.50, ">>>": 1.00}

    # prepare log returns for autoscale if requested
    use_auto = ("自動" in str(rel_mode))
    ret_log = None
    if use_auto:
        if prices_train is not None and trading_days_per_year is not None:
            try:
                ret_log = np.log(prices_train[assets]).diff().dropna()
            except Exception:
                ret_log = None

    for rv in rel_views or []:
        a = str(rv.get("a", ""))
        b = str(rv.get("b", ""))
        if (a not in idx) or (b not in idx) or (a == b):
            continue
        strength = str(rv.get("strength", ">"))
        conf = float(rv.get("confidence", 0.6))
        conf = float(np.clip(conf, 0.0, 1.0))

        row = np.zeros(n, dtype=float)
        row[idx[a]] = 1.0
        row[idx[b]] = -1.0
        P_rows.append(row)

        if use_auto and ret_log is not None and a in ret_log.columns and b in ret_log.columns:
            spread = (ret_log[a] - ret_log[b]).dropna()
            if spread.shape[0] >= 20:
                sigma = float(spread.std(ddof=1)) * float(np.sqrt(trading_days_per_year))
            else:
                sigma = 0.02  # fallback annual diff
            k = auto_k.get(strength, 0.25)
            q = float(rel_q_base) * float(k) * float(max(sigma, 1e-6))
        else:
            q0 = fixed_map.get(strength, 0.005)
            q = float(rel_q_base) * float(q0)

        Q_vals.append(float(q))
        confs.append(conf if conf > 0 else 0.05)

    if not P_rows:
        return pi

    P = np.vstack(P_rows)
    Q = np.asarray(Q_vals, dtype=float)

    # omega: diagonal; higher confidence => smaller omega
    # base variance of each view: diag(P * (tau*S) * P^T)
    try:
        base = np.diag(np.diag(P @ (tau * S.values) @ P.T))
        conf_arr = np.clip(np.asarray(confs, dtype=float), 0.05, 1.0)
        omega = base / conf_arr  # confidence 0.5 => omega doubled, 1.0 => omega base
        omega = pd.DataFrame(omega)
    except Exception:
        omega = "default"

    bl = black_litterman.BlackLittermanModel(
        S,
        pi=pi,
        P=P,
        Q=Q,
        omega=omega,
        tau=tau,
    )
    mu_post = bl.bl_returns()
    if isinstance(mu_post, pd.DataFrame):
        mu_post = mu_post.iloc[:, 0]
    return pd.Series(mu_post, index=assets)
