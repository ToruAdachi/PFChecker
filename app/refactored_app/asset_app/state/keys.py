# -*- coding: utf-8 -*-
"""Centralized Streamlit session_state keys.

Keeping keys in one place reduces KeyError risk and makes refactoring easier.
"""

# Portfolio
PORTFOLIO_ITEMS = "portfolio_items"
PORTFOLIO_WEIGHTS = "portfolio_weights"

# Portfolio UI helpers
OVERWRITE_ARMED = "_overwrite_armed"
DELETE_ARMED = "_delete_armed"
PF_DELETE_TARGET_ID = "pf_delete_target_id"
PF_DELETE_RESET = "_pf_delete_reset"

NEW_DISPLAY_NAME = "new_display_name"
NEW_TICKER = "new_ticker"
NEW_ASSET_TYPE = "new_asset_type"
NEW_CURRENCY = "new_currency"
PF_NEW_ASSET_TYPE_UI = "pf_new_asset_type_ui"
PF_NEW_CURRENCY_UI = "pf_new_currency_ui"

PF_VIEW_ASSETS = "_pf_view_assets"

# Optimization
OPT_CHOICES_PERSIST = "opt_choices_persist"
OPT_CHOICES_UI = "opt_choices_ui"

# Black-Litterman
BL_SETTINGS_PERSIST = "bl_settings_persist"

BL_MARKET_MODEL = "_bl_market_model"
BL_DELTA = "_bl_delta"
BL_ALPHA = "_bl_alpha"
BL_TAU = "_bl_tau"
BL_VIEWS = "_bl_views"
BL_VIEWS_CONF = "_bl_views_conf"
BL_REL_VIEWS = "_bl_rel_views"
BL_REL_MODE = "_bl_rel_mode"
BL_REL_Q_BASE = "_bl_rel_q_base"

BL_MARKET_MODEL_UI = "bl_market_model_ui"
BL_RISK_PROFILE_UI = "bl_risk_profile_ui"
BL_ALPHA_UI = "bl_alpha_ui"
BL_USE_VIEWS_UI = "bl_use_views_ui"
BL_TAU_UI = "bl_tau_ui"

# Cached data
PRICES_JPY_CACHE = "_prices_jpy_cache"
PRICES_JPY_CACHE_COLS = "_prices_jpy_cache_cols"


def weight_key(display_name: str) -> str:
    """Return session_state key for a weight widget."""
    return f"w_{display_name}"
