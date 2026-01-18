# -*- coding: utf-8 -*-
"""Path helpers.

This project often relies on local external files (e.g., data_j.xls, csv/json caches).

Deployment convention (recommended):
  - Application code lives under: /home/pi/refactored_app_current/
  - External files live one directory ABOVE the app directory: /home/pi/

The helper functions below implement that convention while keeping the code portable.
"""

from __future__ import annotations

from pathlib import Path
import os


def project_dir() -> Path:
    """Return the directory that contains app.py (i.e., the app folder)."""
    # Prefer the current working directory when it looks like the project root.
    # This makes the "external data dir = one level above the app dir" convention
    # stable even when the code is deployed via a symlink (refactored_app_current)
    # pointing to a real release directory.
    cwd = Path.cwd()
    if (cwd / "app.py").exists() and (cwd / "asset_app").exists():
        return cwd.resolve()

    # Fallback: asset_app/config/paths.py -> asset_app -> <project_dir>
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the external data directory (one level above the app folder).

    You can override this location with the environment variable ASSET_APP_DATA_DIR.
    """
    env = os.environ.get("ASSET_APP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_dir().parent


def in_data_dir(filename: str) -> Path:
    return data_dir() / filename


def default_portfolios_path() -> Path:
    return in_data_dir("portfolios.json")


def default_jpx_cache_path() -> Path:
    return in_data_dir("jpx_listings_cache.csv")


def default_instruments_path() -> Path:
    return in_data_dir("instruments.csv")


def default_us_tickers_path() -> Path:
    """Optional local U.S. ticker master CSV path."""
    return in_data_dir("us_tickers.csv")


def default_us_mcap_cache_path() -> Path:
    """Optional local market-cap cache for ranking/search.

    This file is populated opportunistically when the app fetches market caps from Yahoo.
    """
    return in_data_dir("us_mcap_cache.json")


def default_us_meta_cache_path() -> Path:
    """Optional local metadata cache for ranking/search.

    If present, it may contain fields like sector/industry/sp500 membership.
    """
    return in_data_dir("us_meta_cache.json")
