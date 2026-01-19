# -*- coding: utf-8 -*-
"""Portfolio simulation + metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.constants import TAX_RATE_DEFAULT
from ..services.optimizer import optimize_weights_cached

def build_buy_and_hold_pf(
    prices_jpy: pd.DataFrame, weights: pd.Series, initial_value: float
):
    w = weights.reindex(prices_jpy.columns).fillna(0.0)
    w_sum = float(w.sum())
    if w_sum > 0:
        w = w / w_sum
    elif len(prices_jpy.columns) > 0:
        w = pd.Series(1.0 / float(len(prices_jpy.columns)), index=prices_jpy.columns)
    p0 = prices_jpy.iloc[0]
    shares = (initial_value * w) / p0
    pf_value = (prices_jpy * shares).sum(axis=1)
    pf_index = 100.0 * (pf_value / pf_value.iloc[0])
    return shares, pf_value.rename("PF_VALUE"), pf_index.rename("PF_INDEX")


def calc_metrics_from_pf(
    pf_value: pd.Series, trading_days_per_year: float, rf_annual: float = 0.0
):
    log_ret = np.log(pf_value).diff().dropna()
    rf_daily = pd.Series(rf_annual / trading_days_per_year, index=log_ret.index)
    excess = log_ret - rf_daily

    ann_ret = log_ret.mean() * trading_days_per_year
    ann_vol = log_ret.std(ddof=1) * np.sqrt(trading_days_per_year)

    running_max = pf_value.cummax()
    dd = pf_value / running_max - 1.0
    mdd = dd.min()

    sharpe = np.nan
    if excess.std(ddof=1) > 0:
        sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(trading_days_per_year)

    cum_ret = (pf_value.iloc[-1] / pf_value.iloc[0]) - 1.0

    metrics = {
        "Cumulative Return": float(cum_ret),
        "Annualized Return (log)": float(ann_ret),
        "Annualized Volatility": float(ann_vol),
        "Max Drawdown": float(mdd),
        "Sharpe (annual)": float(sharpe),
    }
    return metrics, log_ret, dd


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
    if freq.upper() == "D":
        return set(idx)

    s = pd.Series(1, index=idx)
    if freq.upper() == "W":
        reb = s.groupby([idx.isocalendar().year, idx.isocalendar().week]).tail(1).index
        return set(pd.DatetimeIndex(reb))
    if freq.upper() == "M":
        reb = s.groupby([idx.year, idx.month]).tail(1).index
        return set(pd.DatetimeIndex(reb))
    if freq.upper() == "Y":
        reb = s.groupby(idx.year).tail(1).index
        return set(pd.DatetimeIndex(reb))
    raise ValueError("freq must be D/W/M/Y")


def build_rebalanced_pf_with_tax(
    prices_jpy: pd.DataFrame,
    target_weights: pd.Series,
    initial_value: float,
    rebalance_freq: str = "D",
    tax_rate: float = TAX_RATE_DEFAULT,
    min_target_for_rel: float | None = None,
    **_ignored_kwargs,
):
    """
    カレンダー（D/W/M/Y）でリバランス + 売却益課税（平均原価・損益通算なし）

    Note:
      - min_target_for_rel は、旧UI/旧コードとの互換のために受け取りますが、この関数では使用しません。
    """
    prices = prices_jpy.dropna(how="any").copy()
    tickers = list(prices.columns)

    w = target_weights.reindex(tickers).fillna(0.0).astype(float)
    w_sum = float(w.sum())
    if w_sum > 0:
        w = w / w_sum
    elif len(tickers) > 0:
        w = pd.Series(1.0 / float(len(tickers)), index=tickers)

    rebalance_dates = get_rebalance_dates(prices.index, rebalance_freq)

    p0 = prices.iloc[0]
    shares = (initial_value * w) / p0
    cash = 0.0
    avg_cost = p0.copy()

    pf_values = []
    tax_paid = []

    for dt, px in prices.iterrows():
        pos_value = shares * px
        total_value = float(pos_value.sum() + cash)

        day_tax = 0.0

        if dt in rebalance_dates:
            target_value = total_value * w
            target_shares = target_value / px
            delta = target_shares - shares

            # 売り（利益に課税）
            sell_mask = delta < 0
            if sell_mask.any():
                sell_qty = (-delta[sell_mask]).copy()
                sell_price = px[sell_mask]
                proceeds = float((sell_qty * sell_price).sum())

                realized_gain = float(
                    ((sell_price - avg_cost[sell_mask]) * sell_qty).sum()
                )
                tax = tax_rate * max(realized_gain, 0.0)
                day_tax += tax

                shares.loc[sell_mask] -= sell_qty
                cash += proceeds
                cash -= tax

            # 買い（キャッシュの範囲）
            buy_mask = delta > 0
            if buy_mask.any():
                buy_qty_desired = delta[buy_mask].copy()
                buy_price = px[buy_mask]
                cost_desired = float((buy_qty_desired * buy_price).sum())

                if cost_desired > 0:
                    scale = 1.0
                    if cash < cost_desired:
                        scale = max(cash / cost_desired, 0.0)

                    buy_qty = buy_qty_desired * scale
                    cost = float((buy_qty * buy_price).sum())

                    old_sh = shares[buy_mask]
                    old_cost = avg_cost[buy_mask]
                    new_sh = old_sh + buy_qty
                    avg_cost_update = (
                        old_cost * old_sh + buy_price * buy_qty
                    ) / new_sh.replace(0, np.nan)
                    avg_cost.loc[buy_mask] = avg_cost_update.fillna(old_cost)

                    shares.loc[buy_mask] += buy_qty
                    cash -= cost

            pos_value = shares * px
            total_value = float(pos_value.sum() + cash)

        pf_values.append(total_value)
        tax_paid.append(day_tax)

    pf_value = pd.Series(
        pf_values, index=prices.index, name=f"PF_VALUE_REBAL_{rebalance_freq}_TAX"
    )
    pf_index = (100.0 * (pf_value / pf_value.iloc[0])).rename(
        f"PF_INDEX_REBAL_{rebalance_freq}_TAX"
    )
    tax_paid_s = pd.Series(
        tax_paid, index=prices.index, name=f"TAX_PAID_{rebalance_freq}"
    )
    return pf_value, pf_index, tax_paid_s


def build_threshold_rebalanced_pf_with_tax(
    prices_jpy: pd.DataFrame,
    target_weights: pd.Series,
    initial_value: float,
    threshold_rel: float = 0.10,  # 例：0.10 = 目標比率から±10%相対ズレで発火
    cooldown_days: int = 20,
    tax_rate: float = TAX_RATE_DEFAULT,
    min_target_for_rel: float = 0.01,
    eps: float = 1e-12,
):
    """
    相対ズレ（目標比率に対する割合ズレ）でリバランス + 売却益課税（平均原価・損益通算なし）
    - 毎日、現在比率 w_now を算出
    - 相対ズレ: |w_now - w_target| / max(w_target, eps)
    - max相対ズレ >= threshold_rel かつ cooldown を満たすときだけリバランス
    """
    prices = prices_jpy.dropna(how="any").copy()
    tickers = list(prices.columns)

    w_target = target_weights.reindex(tickers).fillna(0.0).astype(float)
    w_sum = float(w_target.sum())
    if w_sum > 0:
        w_target = w_target / w_sum
    elif len(tickers) > 0:
        w_target = pd.Series(1.0 / float(len(tickers)), index=tickers)

    p0 = prices.iloc[0]
    shares = (initial_value * w_target) / p0
    cash = 0.0
    avg_cost = p0.copy()

    pf_values = []
    tax_paid = []
    did_rebalance = []
    max_rel_dev_list = []

    last_reb_i = -(10**9)

    for i, (dt, px) in enumerate(prices.iterrows()):
        pos_value = shares * px
        total_value = float(pos_value.sum() + cash)

        # 判定は総資産（株+cash）ベース
        w_now = (pos_value / total_value).fillna(0.0)

        denom = w_target.clip(lower=min_target_for_rel)
        rel_dev = (
            ((w_now - w_target).abs() / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        max_rel_dev = float(rel_dev.max())
        max_rel_dev_list.append(max_rel_dev)

        ok_cooldown = (cooldown_days <= 0) or ((i - last_reb_i) >= cooldown_days)

        day_tax = 0.0
        reb = False

        if ok_cooldown and (max_rel_dev >= threshold_rel):
            reb = True

            target_value = total_value * w_target
            target_shares = target_value / px
            delta = target_shares - shares

            # 売り
            sell_mask = delta < 0
            if sell_mask.any():
                sell_qty = (-delta[sell_mask]).copy()
                sell_price = px[sell_mask]
                proceeds = float((sell_qty * sell_price).sum())

                realized_gain = float(
                    ((sell_price - avg_cost[sell_mask]) * sell_qty).sum()
                )
                tax = tax_rate * max(realized_gain, 0.0)
                day_tax += tax

                shares.loc[sell_mask] -= sell_qty
                cash += proceeds
                cash -= tax

            # 買い
            buy_mask = delta > 0
            if buy_mask.any():
                buy_qty_desired = delta[buy_mask].copy()
                buy_price = px[buy_mask]
                cost_desired = float((buy_qty_desired * buy_price).sum())
                if cost_desired > 0:
                    scale = 1.0
                    if cash < cost_desired:
                        scale = max(cash / cost_desired, 0.0)

                    buy_qty = buy_qty_desired * scale
                    cost = float((buy_qty * buy_price).sum())

                    old_sh = shares[buy_mask]
                    old_cost = avg_cost[buy_mask]
                    new_sh = old_sh + buy_qty
                    avg_cost_update = (
                        old_cost * old_sh + buy_price * buy_qty
                    ) / new_sh.replace(0, np.nan)
                    avg_cost.loc[buy_mask] = avg_cost_update.fillna(old_cost)

                    shares.loc[buy_mask] += buy_qty
                    cash -= cost

            last_reb_i = i

        pf_values.append(float((shares * px).sum() + cash))
        tax_paid.append(day_tax)
        did_rebalance.append(1 if reb else 0)

    pf_value = pd.Series(pf_values, index=prices.index, name="PF_VALUE_THR_REL_TAX")
    pf_index = (100.0 * (pf_value / pf_value.iloc[0])).rename("PF_INDEX_THR_REL_TAX")
    tax_paid_s = pd.Series(tax_paid, index=prices.index, name="TAX_PAID_THR_REL")
    reb_s = pd.Series(did_rebalance, index=prices.index, name="REBALANCE_FLAG")
    max_rel_dev_s = pd.Series(
        max_rel_dev_list, index=prices.index, name="MAX_REL_WEIGHT_DEV"
    )
    return pf_value, pf_index, tax_paid_s, reb_s, max_rel_dev_s


# =========================
# Optimization helpers
# =========================



def build_walk_forward_pf_with_tax(
    prices_eval: pd.DataFrame,
    prices_all_for_train: pd.DataFrame,
    initial_value: float,
    opt_freq: str,  # 目標更新頻度（最適化頻度）: "W"/"M"/"Y"/"D"
    method: str,
    rf_annual: float,
    trading_days_per_year: float,
    bl_settings: tuple | None,
    lookback_years: int,
    tax_rate: float,
    apply_tax: bool,
    threshold_rel: float = 0.10,
    cooldown_days: int = 20,
    min_target_for_rel: float = 0.01,
    eps: float = 1e-12,
):
    """
    ウォークフォワード（条件リバランス版）
    - opt_freq の日に学習→最適化し、目標ウェイト w_target を更新
    - 売買（リバランス）は毎日：
        max(|w_now - w_target| / max(w_target, eps)) >= threshold_rel
        かつ cooldown_days 経過
    """
    prices = prices_eval.dropna(how="any").copy()
    tickers = list(prices.columns)

    opt_dates = get_rebalance_dates(prices.index, opt_freq)

    # 初期：開始日前の直近lookbackで最適化して初期配分
    dt0 = prices.index[0]
    px0 = prices.loc[dt0]

    train_end0 = dt0 - pd.Timedelta(days=1)
    train_start0 = train_end0 - pd.DateOffset(years=lookback_years)

    train0 = (
        prices_all_for_train.loc[
            (prices_all_for_train.index >= train_start0)
            & (prices_all_for_train.index <= train_end0),
            tickers,
        ]
        .copy()
        .dropna(how="any")
    )

    if train0.shape[0] < 2:
        raise ValueError(
            "WF: 学習データが不足しています（開始日より前に十分な価格がありません）。"
        )

    w_target = (
        optimize_weights_cached(train0, method, rf_annual, trading_days_per_year, bl_settings=bl_settings)
        .reindex(tickers)
        .fillna(0.0)
    )
    w_target = w_target / w_target.sum()

    shares = (initial_value * w_target) / px0
    cash = 0.0
    avg_cost = px0.copy()

    pf_values = []
    tax_paid = []
    weights_hist = []  # (date, w_target)
    rebal_flags = []
    max_rel_devs = []

    last_reb_i = -(10**9)

    for i, (dt, px) in enumerate(prices.iterrows()):
        # 1) 目標更新（最適化日）
        if dt in opt_dates:
            train_end = dt - pd.Timedelta(days=1)
            train_start = train_end - pd.DateOffset(years=lookback_years)

            train = (
                prices_all_for_train.loc[
                    (prices_all_for_train.index >= train_start)
                    & (prices_all_for_train.index <= train_end),
                    tickers,
                ]
                .copy()
                .dropna(how="any")
            )

            if train.shape[0] >= 2:
                w_new = (
                    optimize_weights_cached(train, method, rf_annual, trading_days_per_year, bl_settings=bl_settings)
                    .reindex(tickers)
                    .fillna(0.0)
                )
                w_new = w_new / w_new.sum()
                w_target = w_new
                weights_hist.append((dt, w_target))

        # 2) 条件判定（毎日）
        pos_value = shares * px
        total_value = float(pos_value.sum() + cash)

        w_now = (pos_value / total_value).fillna(0.0)

        denom = w_target.clip(lower=min_target_for_rel)
        rel_dev = (
            ((w_now - w_target).abs() / denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        max_rel_dev = float(rel_dev.max())
        max_rel_devs.append(max_rel_dev)

        ok_cooldown = (cooldown_days <= 0) or ((i - last_reb_i) >= cooldown_days)

        day_tax = 0.0
        did_reb = 0

        if ok_cooldown and (max_rel_dev >= threshold_rel):
            did_reb = 1

            target_value = total_value * w_target
            target_shares = target_value / px
            delta = target_shares - shares

            # 売り
            sell_mask = delta < 0
            if sell_mask.any():
                sell_qty = (-delta[sell_mask]).copy()
                sell_price = px[sell_mask]
                proceeds = float((sell_qty * sell_price).sum())

                realized_gain = float(
                    ((sell_price - avg_cost[sell_mask]) * sell_qty).sum()
                )
                tax = (tax_rate * max(realized_gain, 0.0)) if apply_tax else 0.0
                day_tax += tax

                shares.loc[sell_mask] -= sell_qty
                cash += proceeds
                cash -= tax

            # 買い
            buy_mask = delta > 0
            if buy_mask.any():
                buy_qty_desired = delta[buy_mask].copy()
                buy_price = px[buy_mask]
                cost_desired = float((buy_qty_desired * buy_price).sum())
                if cost_desired > 0:
                    scale = 1.0
                    if cash < cost_desired:
                        scale = max(cash / cost_desired, 0.0)
                    buy_qty = buy_qty_desired * scale
                    cost = float((buy_qty * buy_price).sum())

                    old_sh = shares[buy_mask]
                    old_cost = avg_cost[buy_mask]
                    new_sh = old_sh + buy_qty
                    avg_cost_update = (
                        old_cost * old_sh + buy_price * buy_qty
                    ) / new_sh.replace(0, np.nan)
                    avg_cost.loc[buy_mask] = avg_cost_update.fillna(old_cost)

                    shares.loc[buy_mask] += buy_qty
                    cash -= cost

            last_reb_i = i

        pf_values.append(float((shares * px).sum() + cash))
        tax_paid.append(day_tax)
        rebal_flags.append(did_reb)

    pf_value = pd.Series(
        pf_values, index=prices.index, name=f"PF_WF_{method}_{opt_freq}_COND"
    )
    pf_index = (100.0 * (pf_value / pf_value.iloc[0])).rename(
        f"PF_INDEX_WF_{method}_{opt_freq}_COND"
    )
    tax_paid_s = pd.Series(
        tax_paid, index=prices.index, name=f"TAX_PAID_WF_{method}_{opt_freq}_COND"
    )
    rebal_flag_s = pd.Series(rebal_flags, index=prices.index, name="WF_REBALANCE_FLAG")
    max_rel_dev_s = pd.Series(
        max_rel_devs, index=prices.index, name="WF_MAX_REL_WEIGHT_DEV"
    )
    return pf_value, pf_index, tax_paid_s, weights_hist, rebal_flag_s, max_rel_dev_s
