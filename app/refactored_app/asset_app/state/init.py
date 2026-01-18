# -*- coding: utf-8 -*-
"""Session state initialization.

Streamlit reruns the script frequently; any access to st.session_state[...] with a
missing key will raise KeyError. To prevent this, the app calls
ensure_state_defaults(...) early in app.py.
"""

from __future__ import annotations

from typing import MutableMapping, Any

from asset_app.state import keys


def ensure_state_defaults(ss: MutableMapping[str, Any]) -> None:
    """Initialize all required session_state keys if missing.

    This function is intentionally conservative: it only sets defaults when the key
    does not yet exist, and never overwrites user choices.
    """

    defaults: dict[str, Any] = {
        # Portfolio
        keys.PORTFOLIO_ITEMS: [],
        keys.PORTFOLIO_WEIGHTS: {},
        # Portfolio UI
        keys.OVERWRITE_ARMED: None,
        keys.DELETE_ARMED: None,
        keys.PF_DELETE_TARGET_ID: "(未選択)",
        keys.PF_DELETE_RESET: False,
        # New item form
        keys.NEW_DISPLAY_NAME: "",
        keys.NEW_TICKER: "",
        keys.NEW_ASSET_TYPE: "equity",
        keys.NEW_CURRENCY: "JPY",
        keys.PF_NEW_ASSET_TYPE_UI: "equity",
        keys.PF_NEW_CURRENCY_UI: "JPY",
        # View
        keys.PF_VIEW_ASSETS: [],
        # Optimization
        keys.OPT_CHOICES_PERSIST: ["MaxSharpe"],
        keys.OPT_CHOICES_UI: ["MaxSharpe"],
        # BL persistence (UI-friendly)
        keys.BL_SETTINGS_PERSIST: {
            "bl_market_model_ui": "mcap",
            "bl_risk_profile_ui": "medium",
            "bl_alpha_ui": 0.50,
            "bl_use_views_ui": False,
            "bl_tau_ui": 0.05,
        },
        # Caches
        keys.PRICES_JPY_CACHE: None,
        keys.PRICES_JPY_CACHE_COLS: [],
    }

    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v