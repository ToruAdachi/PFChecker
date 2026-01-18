#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a local US ticker dictionary (NASDAQ + major U.S. exchanges) for stable offline search.

Data source: NASDAQ Trader Symbol Directory.

- NASDAQ listed: nasdaqlisted.txt
- Other listed (incl. NYSE/NYSEARCA/NYSEAMERICAN/BATS/IEXG): otherlisted.txt

We include:
  - NASDAQ (from nasdaqlisted)
  - ALL "otherlisted" rows except test issues
    (This is required to cover ETFs like VOO/VTI/EDV on NYSEARCA.)

Additionally we append common crypto symbols for Yahoo Finance (yfinance):
  - BTC-USD (Bitcoin)
  - ETH-USD (Ethereum)

Output CSV columns:
  symbol,name,exchange,asset_class,source,updated_at_utc

Refs (NASDA...):
  https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
  https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt
  https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import requests


NASDAQLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


def _download_text(url: str, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def _parse_pipe_table(text: str):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # Drop trailer lines
    lines = [ln for ln in lines if not ln.startswith("File Creation Time:")]
    header = lines[0].split("|")
    for ln in lines[1:]:
        row = ln.split("|")
        if len(row) != len(header):
            continue
        yield dict(zip(header, row))


def build(out_path: Path) -> None:
    nasdaq_text = _download_text(NASDAQLISTED_URL)
    other_text = _download_text(OTHERLISTED_URL)

    rows: dict[str, dict] = {}
    exch_map = {
        # Ref: NASDAQ Trader Symbol Directory definitions
        "N": "NYSE",
        "A": "NYSEAMERICAN",
        "P": "NYSEARCA",
        "Z": "BATS",
        "V": "IEXG",
    }

    # NASDAQ-listed
    for r in _parse_pipe_table(nasdaq_text):
        sym = (r.get("Symbol") or "").strip()
        if not sym:
            continue
        if (r.get("Test Issue") or "").strip().upper() == "Y":
            continue
        name = (r.get("Security Name") or "").strip()
        rows[sym] = {
            "symbol": sym,
            "name": name,
            "exchange": "NASDAQ",
            "asset_class": "equity",
            "source": "nasdaqlisted",
        }

    # Other listed (NYSE/NYSEARCA/NYSEAMERICAN/etc.)
    for r in _parse_pipe_table(other_text):
        if (r.get("Test Issue") or "").strip().upper() == "Y":
            continue
        exch = (r.get("Exchange") or "").strip().upper()
        sym = (r.get("ACT Symbol") or r.get("Symbol") or "").strip()
        if not sym:
            continue
        name = (r.get("Security Name") or "").strip()
        rows[sym] = {
            "symbol": sym,
            "name": name,
            "exchange": exch_map.get(exch, exch),
            "asset_class": "equity",
            "source": "otherlisted",
        }

    # Append crypto (Yahoo Finance symbols)
    rows["BTC-USD"] = {
        "symbol": "BTC-USD",
        "name": "Bitcoin",
        "exchange": "CRYPTO",
        "asset_class": "crypto",
        "source": "static",
    }
    rows["ETH-USD"] = {
        "symbol": "ETH-USD",
        "name": "Ethereum",
        "exchange": "CRYPTO",
        "asset_class": "crypto",
        "source": "static",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["symbol", "name", "exchange", "asset_class", "source", "updated_at_utc"],
        )
        w.writeheader()
        for sym in sorted(rows.keys()):
            row = dict(rows[sym])
            row["updated_at_utc"] = now
            w.writerow(row)


if __name__ == "__main__":
    # Default output for Raspberry Pi deployment
    build(Path("/home/pi/us_tickers.csv"))
    print("Wrote /home/pi/us_tickers.csv")
