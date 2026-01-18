# -*- coding: utf-8 -*-
"""Portfolio optimizers (PyPortfolioOpt / SciPy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Internal dependencies (kept here to minimize cross-imports in UI)
from asset_app.infra.market_data import fetch_market_caps, get_rep_market_prices_jpy
from asset_app.services.black_litterman import (
    _normalize_weights,
    compute_bl_posterior_returns,
    read_bl_settings,
)

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, black_litterman
    _HAVE_PYPORTFOLIOOPT = True
except Exception:
    EfficientFrontier = risk_models = expected_returns = black_litterman = None
    _HAVE_PYPORTFOLIOOPT = False

try:
    from scipy.optimize import minimize
    _HAVE_SCIPY = True
except Exception:
    minimize = None
    _HAVE_SCIPY = False

def optimize_weights_pyportfolioopt(
    prices_train: pd.DataFrame,
    method: str,
    rf_annual: float,
    trading_days_per_year: float,
    bl_settings: tuple | None = None,
) -> pd.Series:
    """
    PyPortfolioOpt-based optimizer.

    Methods:
      - MaxSharpe / MinVol
      - BL_MaxSharpe / BL_MinVol  (Views optional + blend optional)

    BL notes:
      - We avoid BlackLittermanModel(P,Q) to keep "Viewsなし" valid (no Q required).
      - Prior is computed by market_implied_prior_returns().
      - Posterior uses absolute views (optional) via compute_bl_posterior_returns().
      - Expected returns used in optimization are optionally blended with historical mean:
          mu_used = alpha * mu_bl + (1-alpha) * mu_hist
    """
    # Covariance on PF universe (used for optimization regardless of market model)
    S_pf = risk_models.sample_cov(prices_train, frequency=trading_days_per_year)

    # ---- Black-Litterman path ----
    if method in ("BL_MaxSharpe", "BL_MinVol"):
        bl_alpha, bl_tau, bl_abs_views, bl_abs_confs, bl_rel_views, bl_rel_mode, bl_rel_q_base, bl_market_model, bl_delta = read_bl_settings()
        pf_tickers = tuple(prices_train.columns.tolist())

        # --- Build prior (pi) ---
        if bl_market_model == "代表ETF市場（資産クラス）":
            # 1) Representative market universe prior
            rep_prices_jpy = get_rep_market_prices_jpy(
                start=str(prices_train.index.min().date()),
                end=str(prices_train.index.max().date()),
            ).dropna(how="any")

            if rep_prices_jpy.shape[0] < 60:
                raise ValueError("代表ETF市場のデータが不足しています（最低60営業日程度必要）。")

            rep_tickers = tuple(rep_prices_jpy.columns.tolist())
            S_mkt = risk_models.sample_cov(rep_prices_jpy, frequency=trading_days_per_year)

            mcap_rep = fetch_market_caps(rep_tickers)
            mcap_rep = pd.Series(mcap_rep, index=list(rep_tickers), dtype=float)

            if mcap_rep.isna().all() or float(mcap_rep.fillna(0.0).sum()) <= 0:
                mcap_rep = pd.Series(1.0, index=list(rep_tickers), name="market_cap_proxy")
            else:
                med = float(mcap_rep.dropna().median()) if not mcap_rep.dropna().empty else 1.0
                mcap_rep = mcap_rep.fillna(med)

            pi_mkt = black_litterman.market_implied_prior_returns(mcap_rep, bl_delta, S_mkt)

            # 2) Map market prior -> PF prior via regression on daily log returns
            ret_pf = np.log(prices_train).diff().dropna()
            ret_mkt = np.log(rep_prices_jpy).diff().dropna()
            common = ret_pf.index.intersection(ret_mkt.index)
            ret_pf = ret_pf.loc[common]
            ret_mkt = ret_mkt.loc[common]

            if ret_pf.shape[0] < 60:
                raise ValueError("代表ETF市場への写像に十分な重複データがありません（最低60営業日程度必要）。")

            # X with intercept
            X = np.column_stack([np.ones(len(ret_mkt)), ret_mkt.values])

            # annual -> daily to match regression space
            pi_mkt_daily = (pi_mkt / trading_days_per_year).reindex(ret_mkt.columns)

            pi_pf = {}
            for col in ret_pf.columns:
                y = ret_pf[col].values
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [alpha, betas...]
                a = float(beta[0])
                b = beta[1:]
                mu_d = a + float(np.dot(b, pi_mkt_daily.values))
                pi_pf[col] = mu_d * trading_days_per_year

            pi = pd.Series(pi_pf).reindex(prices_train.columns)

        else:
            # PF-universe market-cap/AUM proxy prior
            mcap = fetch_market_caps(pf_tickers)
            mcap = pd.Series(mcap, index=prices_train.columns, dtype=float)

            if mcap.isna().all() or float(mcap.fillna(0.0).sum()) <= 0:
                mcap = pd.Series(1.0, index=prices_train.columns, name="market_cap_proxy")
            else:
                med = float(mcap.dropna().median()) if not mcap.dropna().empty else 1.0
                mcap = mcap.fillna(med)

            pi = black_litterman.market_implied_prior_returns(mcap, bl_delta, S_pf)



        # --- Sensitivity: how implied prior returns π change with δ (risk aversion) ---
        # We store the table in session_state so the main UI can visualize it.
        try:
            _delta_grid = [
                ("積極（δ=1.5）", 1.5),
                ("標準（δ=3.0）", 3.0),
                ("保守（δ=6.0）", 6.0),
            ]

            def _pi_from_mcap_cov(_mcap: pd.Series, _delta: float, _S: pd.DataFrame) -> pd.Series:
                _w = (_mcap / float(_mcap.sum())).astype(float)
                _base = _S.dot(_w)
                return (_delta * _base).astype(float)

            if bl_market_model == "代表ETF市場（資産クラス）":
                # Build regression coefficients once, then recompute mapped π for each δ.
                _coef = {}
                for _col in ret_pf.columns:
                    _y = ret_pf[_col].values
                    _beta, *_ = np.linalg.lstsq(X, _y, rcond=None)
                    _coef[_col] = _beta  # [alpha, betas...]

                # Store regression betas for visualization (stocks x rep ETFs)
                try:
                    _betas_mat = {}
                    _alpha_vec = {}
                    for _col, _beta in _coef.items():
                        _alpha_vec[_col] = float(_beta[0])
                        _betas_mat[_col] = pd.Series(_beta[1:], index=ret_mkt.columns, dtype="float64")
                    _betas_df = pd.DataFrame(_betas_mat).T.reindex(index=prices_train.columns).reindex(columns=ret_mkt.columns)
                    st.session_state["_bl_rep_betas"] = {
                        "market_model": bl_market_model,
                        "assets": list(_betas_df.index),
                        "etfs": list(_betas_df.columns),
                        "betas": _betas_df,
                        "alpha": pd.Series(_alpha_vec).reindex(index=prices_train.columns),
                    }
                except Exception:
                    pass

                _pi_pf_by_delta = {}
                for _lab, _dlt in _delta_grid:
                    _pi_mkt = _pi_from_mcap_cov(mcap_rep, float(_dlt), S_mkt).reindex(ret_mkt.columns)
                    _pi_mkt_daily = (_pi_mkt / trading_days_per_year).reindex(ret_mkt.columns)

                    _pi_pf_tmp = {}
                    for _col in ret_pf.columns:
                        _beta = _coef[_col]
                        _a = float(_beta[0])
                        _b = _beta[1:]
                        _mu_d = _a + float(np.dot(_b, _pi_mkt_daily.values))
                        _pi_pf_tmp[_col] = _mu_d * trading_days_per_year

                    _pi_pf_by_delta[_lab] = pd.Series(_pi_pf_tmp).reindex(prices_train.columns)

                _pi_sens_df = pd.DataFrame(_pi_pf_by_delta).reindex(prices_train.columns)
            else:
                # PF-universe prior (purely proportional to δ)
                _pi_sens = {}
                for _lab, _dlt in _delta_grid:
                    _pi_sens[_lab] = _pi_from_mcap_cov(mcap, float(_dlt), S_pf).reindex(prices_train.columns)
                _pi_sens_df = pd.DataFrame(_pi_sens).reindex(prices_train.columns)

            st.session_state["_bl_pi_sensitivity"] = {
                "market_model": bl_market_model,
                "assets": list(_pi_sens_df.index),
                "pi": _pi_sens_df,
                "delta_current": float(bl_delta),
            }
        except Exception:
            pass
# Posterior (absolute views optional)
        mu_bl = compute_bl_posterior_returns(pi=pi, S=S_pf, tau=bl_tau, prices_train=prices_train, trading_days_per_year=trading_days_per_year, abs_views=bl_abs_views, abs_view_confs=bl_abs_confs, rel_views=bl_rel_views, rel_mode=bl_rel_mode, rel_q_base=bl_rel_q_base)

        # Blend with historical mean (optional)
        mu_hist = expected_returns.mean_historical_return(prices_train, frequency=trading_days_per_year)
        mu_used = bl_alpha * mu_bl + (1.0 - bl_alpha) * mu_hist

        if method == "BL_MinVol":
            ef = EfficientFrontier(None, S_pf, weight_bounds=(0.0, 1.0))
            ef.min_volatility()
        else:
            ef = EfficientFrontier(mu_used, S_pf, weight_bounds=(0.0, 1.0))
            ef.max_sharpe(risk_free_rate=rf_annual)

        w = ef.clean_weights()
        return _normalize_weights(pd.Series(w))

    # ---- Classic mean-variance (historical mean) ----
    mu = expected_returns.mean_historical_return(prices_train, frequency=trading_days_per_year)
    ef = EfficientFrontier(mu, S_pf, weight_bounds=(0.0, 1.0))

    if method == "MaxSharpe":
        ef.max_sharpe(risk_free_rate=rf_annual)
    elif method == "MinVol":
        ef.min_volatility()
    else:
        raise ValueError("Unknown method")

    w = ef.clean_weights()
    return _normalize_weights(pd.Series(w))
def optimize_weights_scipy(
    prices_train: pd.DataFrame,
    method: str,
    rf_annual: float,
    trading_days_per_year: float,
) -> pd.Series:
    log_ret = np.log(prices_train).diff().dropna()
    mu_d = log_ret.mean().values
    cov_d = log_ret.cov().values

    mu = mu_d * trading_days_per_year
    cov = cov_d * trading_days_per_year

    n = prices_train.shape[1]
    x0 = np.ones(n) / n

    def port_vol(w):
        return np.sqrt(float(w.T @ cov @ w))

    if method == "MinVol":

        def obj(w):
            return port_vol(w)

    elif method == "MaxSharpe":

        def obj(w):
            vol = port_vol(w)
            if vol <= 0:
                return 1e9
            ret = float(mu @ w)
            sharpe = (ret - rf_annual) / vol
            return -sharpe

    else:
        raise ValueError("Unknown method")

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n

    res = minimize(obj, x0, bounds=bounds, constraints=cons, method="SLSQP")
    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    return _normalize_weights(pd.Series(res.x, index=prices_train.columns))


def optimize_weights(
    prices_train: pd.DataFrame,
    method: str,
    rf_annual: float,
    trading_days_per_year: float,
) -> pd.Series:
    if _HAVE_PYPORTFOLIOOPT:
        return optimize_weights_pyportfolioopt(
            prices_train, method, rf_annual, trading_days_per_year
        )
    if _HAVE_SCIPY:
        return optimize_weights_scipy(
            prices_train, method, rf_annual, trading_days_per_year
        )
    raise RuntimeError(
        "Optimization requires PyPortfolioOpt or scipy. Please install one of them."
    )




@st.cache_data(ttl=3600, show_spinner=False)
def optimize_weights_cached(
    prices_train: pd.DataFrame,
    method: str,
    rf_annual: float,
    trading_days_per_year: float,
    bl_settings: tuple | None,
) -> pd.Series:
    """Cached optimizer to avoid recomputation on unrelated UI changes.

    Cache key includes prices_train and all parameters that materially affect optimization.
    """
    # Use the same dispatch logic but pass through BL settings explicitly.
    if _HAVE_PYPORTFOLIOOPT:
        return optimize_weights_pyportfolioopt(
            prices_train, method, rf_annual, trading_days_per_year, bl_settings=bl_settings
        )
    if _HAVE_SCIPY:
        # SciPy path does not use BL settings (keep signature consistent)
        return optimize_weights_scipy(
            prices_train, method, rf_annual, trading_days_per_year
        )
    raise RuntimeError(
        "Optimization requires PyPortfolioOpt or scipy. Please install one of them."
    )
