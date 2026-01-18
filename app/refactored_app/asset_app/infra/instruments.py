# -*- coding: utf-8 -*-
"""Instrument master + symbol search."""

from __future__ import annotations

import re
import pandas as pd
import streamlit as st
import yfinance as yf

import requests
import json
from pathlib import Path
from difflib import SequenceMatcher


class YahooRateLimitError(RuntimeError):
    """Raised when Yahoo Finance symbol search is rate-limited (HTTP 429)."""

    def __init__(self, message: str = "Yahoo Finance search rate-limited", status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code

# =========================
# Data helpers
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def load_instruments(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"display_name", "ticker", "asset_type", "currency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"instruments.csv missing columns: {missing}")
    df["display_name"] = df["display_name"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["asset_type"] = df["asset_type"].astype(str)
    df["currency"] = df["currency"].astype(str).str.upper()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def yfinance_symbol_search(query: str, max_results: int = 10) -> list[dict]:
    """Search symbols via Yahoo Finance search endpoint.

    This is used primarily for non-JPX symbols (e.g., U.S. stocks).

    Returns a list of dicts with keys:
      - symbol, name, quoteType, exchange, currency

    Raises:
      - YahooRateLimitError when HTTP 429 is returned.

    Notes:
      - We require 2+ chars in general to reduce call volume.
      - We allow a 1-character alnum query to support tickers like 'F'.
    """
    q = (query or "").strip()
    if len(q) < 2:
        if not (len(q) == 1 and q.isalnum()):
            return []

    # Use Yahoo Finance search API directly for predictable error handling.
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    params = {
        "q": q,
        "quotesCount": int(max_results),
        "newsCount": 0,
        "enableFuzzyQuery": False,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=7)
    except Exception:
        # Network errors -> treat as no candidates.
        return []

    if resp.status_code == 429:
        raise YahooRateLimitError("Yahoo Finance search returned HTTP 429")
    if not resp.ok:
        return []

    try:
        payload = resp.json()
    except Exception:
        return []

    quotes = payload.get("quotes") or []
    out: list[dict] = []
    for it in quotes:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol") or "").strip()
        if not sym:
            continue
        out.append(
            {
                "symbol": sym,
                "name": str(it.get("shortname") or it.get("longname") or it.get("name") or "").strip(),
                "quoteType": str(it.get("quoteType") or "").strip(),
                "exchange": str(it.get("exchange") or it.get("exchDisp") or "").strip(),
                "currency": str(it.get("currency") or "").strip(),
            }
        )
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def load_us_tickers(csv_path: str) -> pd.DataFrame:
    """Load a local U.S. ticker master CSV.

    Expected columns (case-insensitive; extra columns allowed):
      - symbol (required)
      - name (optional)
      - exchange (optional)
      - asset_class (optional; e.g., equity, crypto)

    This file is optional; when missing, callers should fall back gracefully.
    """
    df = pd.read_csv(csv_path)
    # Normalize columns
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError("us_tickers.csv missing required column: symbol")
    if "name" not in df.columns:
        df["name"] = ""
    if "exchange" not in df.columns:
        df["exchange"] = ""
    if "asset_class" not in df.columns:
        df["asset_class"] = "equity"
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["exchange"] = df["exchange"].astype(str).str.strip()
    df["asset_class"] = df["asset_class"].astype(str).str.strip().str.lower()
    df.loc[df["asset_class"] == "", "asset_class"] = "equity"
    df = df[df["symbol"] != ""].copy()

    # --- Bundle-level supplements ---
    # Certain frequently-used instruments (notably ETFs on NYSE Arca) may be absent from
    # the upstream symbol directories used to build us_tickers.csv.
    # To keep the app usable when Yahoo Finance symbol search is rate-limited, we optionally
    # merge a small, curated supplement file that ships with this app.
    try:
        base = Path(__file__).resolve().parents[1]
        extra = base / "data" / "us_tickers_extra.csv"
        if extra.exists():
            df2 = pd.read_csv(extra)
            df2.columns = [str(c).strip().lower() for c in df2.columns]
            if "symbol" in df2.columns:
                if "name" not in df2.columns:
                    df2["name"] = ""
                if "exchange" not in df2.columns:
                    df2["exchange"] = ""
                if "asset_class" not in df2.columns:
                    df2["asset_class"] = "equity"
                df2["symbol"] = df2["symbol"].astype(str).str.strip()
                df2["name"] = df2["name"].astype(str).str.strip()
                df2["exchange"] = df2["exchange"].astype(str).str.strip()
                df2["asset_class"] = df2["asset_class"].astype(str).str.strip().str.lower()
                df2 = df2[df2["symbol"] != ""].copy()
                df = pd.concat([df, df2], ignore_index=True)
                df = df.drop_duplicates(subset=["symbol"], keep="first")
    except Exception:
        pass
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_mcap_cache_json(json_path: str) -> dict[str, float]:
    """Load a simple market-cap cache mapping symbol->market_cap.

    This cache is optional and is populated opportunistically by fetch_market_caps.
    """
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in data.items():
        try:
            kk = str(k).strip().upper()
            if kk:
                out[kk] = float(v)
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_meta_cache_json(json_path: str) -> dict[str, dict]:
    """Load optional metadata cache mapping symbol -> dict.

    This cache is optional. When present it may contain fields like:
      - sector, industry
      - market_cap
      - sp500 (bool)

    The search ranking uses it as an additional, *soft* signal.
    """
    p = Path(json_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in data.items():
        kk = str(k).strip().upper()
        if not kk or not isinstance(v, dict):
            continue
        out[kk] = v
    return out


def _is_subsequence(needle: str, haystack: str) -> bool:
    """Return True if all chars in needle appear in haystack in order."""
    if not needle:
        return True
    it = iter(haystack)
    return all(ch in it for ch in needle)


def search_us_tickers_local(
    df_us: pd.DataFrame,
    query: str,
    limit: int = 30,
    mcap_cache_path: str | None = None,
    meta_cache_path: str | None = None,
) -> list[dict]:
    """Search local U.S. tickers.

    Ranking goals (per user requirements):
      - Prefer candidates that match the *query sequence* (prefix/substring/word-prefix) over unrelated short symbols.
      - Support acronym/initialism queries (e.g., "MS" -> Microsoft) and subsequence queries.
      - Prefer larger market cap when available.
      - Prefer shorter symbols only as a late tie-breaker.

    Notes:
      - Market-cap is optional. If a cache JSON exists (symbol->market_cap), it is used for ranking.
      - Crypto shortcuts (BTC/ETH/BIT...) are treated as aliases to surface BTC-USD/ETH-USD first.
    """
    q = (query or "").strip()
    if not q:
        return []
    q_upper = q.upper()
    q_lower = q.lower()

    # Prefer symbol prefix matches, then name contains.
    sym = df_us["symbol"].astype(str)
    name = df_us.get("name", pd.Series([""] * len(df_us))).astype(str)
    exch = df_us.get("exchange", pd.Series([""] * len(df_us))).astype(str)
    acls = df_us.get("asset_class", pd.Series([""] * len(df_us))).astype(str)

    sym_upper = sym.str.upper()
    acls_lower = acls.astype(str).str.lower()

    # Optional caches
    mcap_map: dict[str, float] = load_mcap_cache_json(mcap_cache_path) if mcap_cache_path else {}
    meta_map: dict[str, dict] = load_meta_cache_json(meta_cache_path) if meta_cache_path else {}

    # Explicit aliases (common crypto shortcuts)
    # NOTE: We intentionally treat prefixes like "BIT" as crypto intent to surface BTC-USD early.
    alias_map = {
        "BTC": "BTC-USD",
        "BITCOIN": "BTC-USD",
        "XBT": "BTC-USD",
        "ETH": "ETH-USD",
        "ETHEREUM": "ETH-USD",
    }
    alias_symbol = alias_map.get(q_upper)
    if alias_symbol is None:
        # Prefix aliases (user may type a partial word)
        if q_upper.startswith("BIT") or q_upper.startswith("BTC") or q_upper.startswith("XBT"):
            alias_symbol = "BTC-USD"
        elif q_upper.startswith("ETH") or q_upper.startswith("ETHER"):
            alias_symbol = "ETH-USD"

    # Basic string matches
    mask_sym_prefix = sym_upper.str.startswith(q_upper)
    mask_sym_contains = sym_upper.str.contains(q_upper, na=False)
    mask_name_prefix = name.str.lower().str.startswith(q_lower)
    mask_name_contains = name.str.lower().str.contains(q_lower, na=False)

    # Word-prefix match: query matches the start of any word in the name
    name_words = name.astype(str).str.lower().str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    mask_name_word_prefix = name_words.str.split().map(lambda ws: any(w.startswith(q_lower) for w in ws if w))

    # Exact symbol match (strongest signal)
    mask_exact = sym_upper.eq(q_upper)

    # Alias match (e.g., query "BTC" should surface BTC-USD first)
    if alias_symbol:
        mask_alias = sym_upper.eq(alias_symbol)
    else:
        mask_alias = pd.Series([False] * len(df_us), index=df_us.index)

    # Candidate set selection.
    # For crypto intent (BTC/ETH or prefixes like "BIT"), aggressively filter to crypto rows
    # (plus alias) to avoid overwhelming the user with unrelated equities like BTCM/BTCS.
    crypto_query = (
        q_upper in {"BTC", "BITCOIN", "XBT", "ETH", "ETHEREUM"}
        or q_upper.startswith("BIT")
        or q_upper.startswith("BTC")
        or q_upper.startswith("XBT")
        or q_upper.startswith("ETH")
        or q_upper.startswith("ETHER")
    )
    if crypto_query:
        df_hit = df_us[mask_alias | acls_lower.eq("crypto")].copy()
    else:
        df_hit = df_us[
            mask_alias
            | mask_exact
            | mask_sym_prefix
            | mask_sym_contains
            | mask_name_prefix
            | mask_name_contains
            | mask_name_word_prefix
        ].copy()
    if df_hit.empty:
        return []

    # Opportunistic market-cap enrichment for better ranking.
    #
    # Rationale:
    # - Nasdaq Trader symbol directories don't include market cap.
    # - Without market cap, very common prefix queries (e.g. "Micro") can rank
    #   unrelated smaller names before mega-caps due to tie-breaks.
    #
    # Policy:
    # - Only when a cache path is provided.
    # - Only for a small set of candidates.
    # - Best-effort; never fails the search.
    # - This does NOT call Yahoo *search* endpoint (the source of HTTP 429 in your case).
    if mcap_cache_path and len(df_hit) <= 200 and len(q_upper) >= 3:
        try:
            top_syms = [str(s).strip().upper() for s in df_hit["symbol"].head(40).tolist()]
            missing = [s for s in top_syms if s and s not in mcap_map]
            if missing:
                from asset_app.infra.market_data import fetch_market_caps

                new_caps = fetch_market_caps(missing)
                if isinstance(new_caps, dict) and new_caps:
                    # Update in-memory map
                    for k, v in new_caps.items():
                        try:
                            kk = str(k).strip().upper()
                            if kk:
                                mcap_map[kk] = float(v)
                        except Exception:
                            continue
                    # Persist to cache path (merge with existing)
                    try:
                        p = Path(mcap_cache_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(json.dumps(mcap_map), encoding="utf-8")
                    except Exception:
                        pass
        except Exception:
            pass

    # --- Scoring ---
    # We compute a composite score emphasizing sequence match, then market cap, then length.
    df_hit["_alias"] = mask_alias.loc[df_hit.index].astype(int)
    df_hit["_exact"] = mask_exact.loc[df_hit.index].astype(int)
    df_hit["_sym_prefix"] = mask_sym_prefix.loc[df_hit.index].astype(int)
    df_hit["_sym_contains"] = mask_sym_contains.loc[df_hit.index].astype(int)
    df_hit["_name_prefix"] = mask_name_prefix.loc[df_hit.index].astype(int)
    df_hit["_name_contains"] = mask_name_contains.loc[df_hit.index].astype(int)
    df_hit["_name_word_prefix"] = mask_name_word_prefix.loc[df_hit.index].astype(int)
    df_hit["_sym_len"] = df_hit["symbol"].astype(str).str.len().astype(int)

    # Acronym / initialism support (e.g., "MS" for Microsoft)
    def _acronym(n: str) -> str:
        parts = re.sub(r"[^A-Za-z0-9 ]+", " ", n).split()
        return "".join(p[0].upper() for p in parts if p)

    df_hit["_acronym"] = df_hit["name"].astype(str).map(_acronym)
    df_hit["_acronym_prefix"] = df_hit["_acronym"].map(lambda a: int(a.startswith(q_upper)))

    # Subsequence match (useful for short queries like "NVI" -> "NVDA")
    df_hit["_subseq_sym"] = df_hit["symbol"].astype(str).str.upper().map(lambda s: int(_is_subsequence(q_upper, s)))
    df_hit["_subseq_name"] = df_hit["name"].astype(str).str.upper().map(lambda s: int(_is_subsequence(q_upper, s[:60])))

    # Fuzzy match ratio against symbol and name (0..1)
    def _ratio(a: str, b: str) -> float:
        try:
            return SequenceMatcher(None, a, b).ratio()
        except Exception:
            return 0.0

    q_u = q_upper
    q_l = q_lower
    df_hit["_fuzzy_sym"] = df_hit["symbol"].astype(str).str.upper().map(lambda s: _ratio(q_u, s))
    df_hit["_fuzzy_name"] = df_hit["name"].astype(str).str.lower().map(lambda s: _ratio(q_l, s[: max(20, len(q_l) + 5)]))

    # Extra fuzzy against the first word of the name (helps distinguish Micro... results)
    def _first_word(n: str) -> str:
        n2 = re.sub(r"[^A-Za-z0-9 ]+", " ", n).strip()
        return (n2.split() or [""])[0]

    df_hit["_name_first"] = df_hit["name"].astype(str).map(_first_word)
    df_hit["_fuzzy_first"] = df_hit["_name_first"].astype(str).str.lower().map(lambda s: _ratio(q_l, s))

    # Market cap (optional)
    def _mcap(sym0: str) -> float:
        try:
            return float(mcap_map.get(str(sym0).strip().upper(), 0.0))
        except Exception:
            return 0.0

    df_hit["_mcap"] = df_hit["symbol"].astype(str).map(_mcap)

    # Metadata-based soft boosts (sector/industry/SP500)
    def _meta(sym0: str, key: str) -> str:
        try:
            return str(meta_map.get(str(sym0).strip().upper(), {}).get(key, ""))
        except Exception:
            return ""

    df_hit["_sector"] = df_hit["symbol"].astype(str).map(lambda s: _meta(s, "sector"))
    df_hit["_industry"] = df_hit["symbol"].astype(str).map(lambda s: _meta(s, "industry"))
    df_hit["_sp500"] = df_hit["symbol"].astype(str).map(lambda s: int(bool(meta_map.get(str(s).strip().upper(), {}).get("sp500", False))))

    q_words = set(re.sub(r"[^a-z0-9 ]+", " ", q_lower).split())
    def _sector_hit(text: str) -> int:
        t = str(text).lower()
        return int(any(w and w in t for w in q_words)) if q_words else 0

    df_hit["_sector_hit"] = df_hit["_sector"].map(_sector_hit)
    df_hit["_industry_hit"] = df_hit["_industry"].map(_sector_hit)

    # No hard-coded company boosts: rely on general ranking signals only.
    df_hit["_pref"] = 0

    # Crypto tie-breaker
    if crypto_query:
        df_hit["_crypto"] = acls_lower.loc[df_hit.index].eq("crypto").astype(int)
    else:
        df_hit["_crypto"] = 0

    # Composite score: prioritize match quality heavily.
    # Composite score: prioritize match quality heavily.
    # IMPORTANT: name-prefix and first-word fuzzy are weighted strongly so that
    #            queries like "Micro" naturally surface "Microsoft" ahead of
    #            less relevant matches when market-cap metadata is missing.
    df_hit["_score"] = (
        1000 * df_hit["_alias"]
        + 900 * df_hit["_exact"]
        + 700 * df_hit["_sym_prefix"]
        + 720 * df_hit["_name_word_prefix"]
        + 650 * df_hit["_name_prefix"]
        + 300 * df_hit["_sym_contains"]
        + 190 * df_hit["_name_contains"]
        + 260 * df_hit["_acronym_prefix"]
        + 220 * df_hit["_subseq_sym"]
        + 120 * df_hit["_subseq_name"]
        + 200 * df_hit["_crypto"]
        + 200 * df_hit["_fuzzy_sym"]
        + 120 * df_hit["_fuzzy_name"]
        + 320 * df_hit["_fuzzy_first"]
        + 60 * df_hit["_sp500"]
        + 40 * df_hit["_sector_hit"]
        + 40 * df_hit["_industry_hit"]
        + df_hit["_pref"]
    )

    # Sort: score desc, market cap desc (when available), then shorter symbol, then alphabetic.
    df_hit = df_hit.sort_values(
        by=["_score", "_mcap", "_sym_len", "symbol"],
        ascending=[False, False, True, True],
    )

    out: list[dict] = []
    for _, r in df_hit.head(limit).iterrows():
        sym_out = str(r.get("symbol") or "").strip()
        acls_out = str(r.get("asset_class") or "").strip().lower()
        # Derive quoteType/currency for better downstream defaults.
        if acls_out == "crypto" or sym_out.upper().endswith("-USD"):
            qt = "CRYPTOCURRENCY"
            cur = "USD"
        else:
            qt = "EQUITY"
            cur = "USD"
        out.append(
            {
                "symbol": sym_out,
                "name": str(r.get("name") or "").strip(),
                "exchange": str(r.get("exchange") or "").strip(),
                "quoteType": qt,
                "currency": cur,
            }
        )
    return out
