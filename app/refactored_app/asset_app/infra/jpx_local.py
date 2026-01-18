# -*- coding: utf-8 -*-
"""JPX local listings loader + search."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import unicodedata

from asset_app.config import paths

# --- JPX listings (for Japanese stock name search) ---
JPX_CACHE_PATH = str(paths.default_jpx_cache_path())

# JPX ローカル辞書（漢字検索用）: 外部ファイルはアプリの1階層上（例: /home/pi）に置く運用を推奨。
JPX_SOURCE_CANDIDATES = ("data_j.xls", "data_j.xlsx", "data_j.csv")


import unicodedata

def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()

def _to_hiragana(text: str) -> str:
    # Convert Katakana to Hiragana (basic range)
    out = []
    for ch in _nfkc(text):
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)

def _to_katakana(text: str) -> str:
    # Convert Hiragana to Katakana (basic range)
    out = []
    for ch in _nfkc(text):
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)

def _normalize_search_text(text: str) -> str:
    return _nfkc(text).lower()

def get_jpx_source_path() -> str | None:
    """Return the first existing JPX master file path in the external data directory."""
    app_dir = paths.data_dir()
    for fn in JPX_SOURCE_CANDIDATES:
        p = app_dir / fn
        if p.exists() and p.is_file():
            return str(p)
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_jpx_source(path: str) -> pd.DataFrame:
    """Load JPX master from local file (xls/xlsx/csv) and normalize columns to code/name/market.

    Notes:
      - .xls requires `xlrd` to be installed in the venv.
      - This function intentionally raises ImportError for missing engines so the UI can show a clear message.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["code", "name", "market"])
    suf = p.suffix.lower()

    if suf == ".xls":
        # pandas needs xlrd for legacy .xls
        try:
            import xlrd  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Reading .xls requires 'xlrd'. Install it with: pip install xlrd"
            ) from e
        df = pd.read_excel(p, engine="xlrd")
    elif suf == ".xlsx":
        df = pd.read_excel(p, engine="openpyxl")
    elif suf == ".csv":
        # try common Japanese encodings
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(p, encoding="cp932")
    else:
        return pd.DataFrame(columns=["code", "name", "market"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "market"])
    out = _normalize_jpx_columns(df)

    # Precompute normalized search fields (for Kana-insensitive search)
    if not out.empty:
        out["name_norm"] = out["name"].map(_normalize_search_text)
        out["name_hira"] = out["name"].map(lambda x: _to_hiragana(_normalize_search_text(x)))
        out["name_kata"] = out["name"].map(lambda x: _to_katakana(_normalize_search_text(x)))
    return out


def _read_uploaded_table(uploaded) -> pd.DataFrame | None:
    """Read an uploaded CSV/XLSX into a DataFrame, trying common encodings."""
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "") or ""
    try:
        if name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded)
        # CSV: try utf-8-sig then shift_jis/cp932
        try:
            return pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(uploaded, encoding="cp932")
    except Exception:
        return None

def _normalize_jpx_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize JPX listing table columns to: code, name, market."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "market"])
    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)
    # Common JPX columns (may vary):
    # コード / 銘柄コード / Code, 銘柄名 / 銘柄名（正式） / Name, 市場・商品区分 / 市場区分 / Market
    code_col = None
    for c in ["コード", "銘柄コード", "Code", "code", "Local Code", "local_code"]:
        if c in df.columns:
            code_col = c
            break
    name_col = None
    for c in ["銘柄名", "銘柄名（正式）", "Name", "name", "銘柄名（English）"]:
        if c in df.columns:
            name_col = c
            break
    market_col = None
    for c in ["市場・商品区分", "市場区分", "Market", "market"]:
        if c in df.columns:
            market_col = c
            break

    out = pd.DataFrame()
    out["code"] = df[code_col] if code_col else ""
    out["name"] = df[name_col] if name_col else ""
    out["market"] = df[market_col] if market_col else ""

    # normalize code to 4-digit string where possible
    out["code"] = out["code"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = out[out["code"].str.fullmatch(r"\d{4}") | out["code"].str.fullmatch(r"\d{5}")].copy()
    out["name"] = out["name"].astype(str).str.strip()
    out["market"] = out["market"].astype(str).str.strip()
    out = out[out["code"].ne("") & out["name"].ne("")].copy()
    out = out.drop_duplicates(subset=["code"], keep="first")
    return out.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_jpx_cache(path: str = JPX_CACHE_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["code", "name", "market"])
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(p, encoding="cp932")
    return _normalize_jpx_columns(df)

def save_jpx_cache(df: pd.DataFrame, path: str = JPX_CACHE_PATH) -> None:
    df2 = _normalize_jpx_columns(df)
    Path(path).write_text(df2.to_csv(index=False), encoding="utf-8-sig")
    load_jpx_cache.clear()

def jpx_local_search(df_jpx: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame:
    """Local search over JPX master (supports Kanji/Kana, code, and ticker-like input).

    The JPX master is expected to have columns: code, name, market.
    If precomputed columns name_norm/name_hira/name_kata exist, they are used for Kana-insensitive search.
    """
    q_raw = (query or "").strip()
    if len(q_raw) < 1 or df_jpx is None or df_jpx.empty:
        return pd.DataFrame(columns=["code", "name", "market"])

    # Accept ticker-like input e.g. 7203.T -> 7203
    q = _normalize_search_text(q_raw)
    q_digits = "".join([c for c in q if c.isdigit()])
    if q_digits and len(q_digits) >= 3 and (q.endswith(".t") or q.endswith(".jp") or q_raw.upper().endswith(".T")):
        q = q_digits

    if q.isdigit():
        m = df_jpx[df_jpx["code"].astype(str).str.startswith(q)]
        return m.head(limit).copy()

    # Kana-insensitive contains match
    q_h = _to_hiragana(q)
    q_k = _to_katakana(q)

    # Ensure search columns exist
    if "name_norm" not in df_jpx.columns:
        name_norm = df_jpx["name"].map(_normalize_search_text)
        name_hira = df_jpx["name"].map(lambda x: _to_hiragana(_normalize_search_text(x)))
        name_kata = df_jpx["name"].map(lambda x: _to_katakana(_normalize_search_text(x)))
    else:
        name_norm = df_jpx["name_norm"]
        name_hira = df_jpx.get("name_hira", df_jpx["name"].map(lambda x: _to_hiragana(_normalize_search_text(x))))
        name_kata = df_jpx.get("name_kata", df_jpx["name"].map(lambda x: _to_katakana(_normalize_search_text(x))))

    mask = name_norm.str.contains(q, regex=False) | name_hira.str.contains(q_h, regex=False) | name_kata.str.contains(q_k, regex=False)
    m = df_jpx[mask]
    return m.head(limit).copy()



def search_jpx_candidates(df_jpx: pd.DataFrame, query: str, limit: int = 20) -> list[dict]:
    """Return JPX candidates as a list of dicts for UI.

    Each candidate dict contains:
      - display_name: str (JPX name)
      - ticker: str (4-digit code + '.T')
      - code: str
      - market: str
      - source: 'JPX'
    """
    out: list[dict] = []
    if df_jpx is None or df_jpx.empty:
        return out
    df = jpx_local_search(df_jpx, query, limit=limit)
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        code = str(r.get('code', '')).strip()
        name = str(r.get('name', '')).strip()
        market = str(r.get('market', '')).strip()
        if not code or not name:
            continue
        tkr = f"{code}.T"
        out.append({
            'display_name': name,
            'ticker': tkr,
            'code': code,
            'market': market,
            'source': 'JPX',
        })
    return out
