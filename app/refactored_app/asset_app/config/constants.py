# -*- coding: utf-8 -*-
"""Project-wide constants."""

TAX_RATE_DEFAULT = 0.20315

# --- Black-Litterman: Representative market universe (asset-class proxies) ---
# Used when BL market model = "代表ETF市場（資産クラス）"
# Note: these are investment proxies, not necessarily in instruments.csv.
REP_MARKET_UNIVERSE = [
    {"display_name": "US Equity (SPY)", "ticker": "SPY", "currency": "USD"},
    {"display_name": "Japan Equity (TOPIX ETF 1306.T)", "ticker": "1306.T", "currency": "JPY"},
    {"display_name": "Dev ex-US (VEA)", "ticker": "VEA", "currency": "USD"},
    {"display_name": "Emerging (VWO)", "ticker": "VWO", "currency": "USD"},
    {"display_name": "US Agg Bond (AGG)", "ticker": "AGG", "currency": "USD"},
    {"display_name": "US Treasury 7-10Y (IEF)", "ticker": "IEF", "currency": "USD"},
    {"display_name": "Gold (GLD)", "ticker": "GLD", "currency": "USD"},
    {"display_name": "Bitcoin (BTC-USD)", "ticker": "BTC-USD", "currency": "USD"},
    {"display_name": "Ethereum (ETH-USD)", "ticker": "ETH-USD", "currency": "USD"},
]
