# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Interactive charts (preferred for mobile)
try:
    import plotly.express as px
    import plotly.graph_objects as go

    def _apply_plotly_mobile_layout(fig, legend_title: str | None = None):
        """Make Plotly figures more readable on mobile by moving legend below the chart.

        Also standardize numeric display (2 decimals) for y-axis ticks and hover.
        """
        if legend_title is not None:
            fig.update_layout(legend_title_text=legend_title)
        fig.update_layout(
            autosize=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="left",
                x=0,
                title=None,
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
            margin=dict(l=10, r=10, t=40, b=70),
        )
        # 2-decimal ticks
        try:
            fig.update_yaxes(tickformat=".2f")
        except Exception:
            pass
        # 2-decimal hover (for time series)
        try:
            fig.update_traces(
                hovertemplate="%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>"
            )
        except Exception:
            pass
        return fig
except Exception:
    px = None
    go = None
import yfinance as yf
from pathlib import Path
import os
import json
import requests
import re

import streamlit.components.v1 as components

# --- Plotly hover-sync renderer (Level-2 UX, frontend-only) ---
#
# Streamlit custom components can be fragile in constrained deployments (routing / assets).
# For Raspberry Pi + systemd, we render the same frontend (Plotly + synchronized table)
# using st.components.v1.html, embedding the local Plotly bundle and the UI logic.
# This avoids /component/* routes entirely, eliminating "trouble loading component" errors.
_PLOTLY_HOVER_SYNC_DIR = (Path(__file__).parent / "asset_app" / "ui" / "plotly_hover_sync")
_PLOTLY_HOVER_SYNC_INDEX = _PLOTLY_HOVER_SYNC_DIR / "index.html"


@st.cache_data(show_spinner=False)
def _load_plotly_hover_sync_index_html(mtime: float) -> str:
    """Load the bundled frontend HTML (includes Plotly bundle inline)."""
    try:
        return _PLOTLY_HOVER_SYNC_INDEX.read_text(encoding="utf-8")
    except Exception:
        return ""


def _render_plotly_hover_sync_front(fig_dict: dict, table_payload: dict | None, height: int, key: str, width: int | None = None):
    """Render Plotly + synced table purely in the frontend (no Streamlit component)."""
    import json as _json

    try:
        _mtime = _PLOTLY_HOVER_SYNC_INDEX.stat().st_mtime
    except Exception:
        _mtime = 0.0
    base = _load_plotly_hover_sync_index_html(_mtime)
    if not base:
        st.error("plotly_hover_sync frontend assets are missing (index.html not found).")
        return

    # The bundled index.html is a Streamlit component frontend that waits for
    # 'streamlit:render' messages. In the st.components.v1.html iframe, those
    # messages are not sent. We therefore strip the message handler tail and
    # call renderFigure(...) directly with injected args.
    marker = "window.addEventListener('message'"
    cut = base.find(marker)
    if cut == -1:
        # Fallback: still try to append a direct render call.
        cut = base.rfind("</script>")
        if cut == -1:
            st.error("plotly_hover_sync frontend template is malformed.")
            return
        head = base[:cut]
        tail = base[cut:]
    else:
        head = base[:cut]
        # We will close the script/body/html ourselves.
        tail = "\n    </script>\n</body>\n</html>\n"

    args = {
        "fig": fig_dict or {},
        "table": table_payload or {},
        "height": int(height),
        "width": int(width) if width is not None else None,
        "_fig_key": str(key),
    }

    injected = (
        "\n    // --- injected initial render (standalone iframe) ---\n"
        f"    const __ARGS__ = {_json.dumps(args, ensure_ascii=False)};\n"
        "    try {\n"
        "      renderFigure(__ARGS__.fig || null, __ARGS__.height || null, __ARGS__.table || null, __ARGS__.width || null);\n"
        "    } catch (e) {\n"
        "      const t = document.getElementById('table');\n"
        "      if (t) { t.innerHTML = '<div style=\"padding:8px;color:#b00020;font-family:system-ui\">Plotly render failed: ' + String(e) + '</div>'; }\n"
        "    }\n"
        "    try { setFrameHeight(); } catch (e) {}\n"
        "\n    // ResizeObserver: keep Plotly responsive when sidebar toggles\n"
        "    try {\n"
        "      const chartDiv = document.getElementById('chart');\n"
        "      const ro = new ResizeObserver(() => {\n"
        "        try { if (window.Plotly && chartDiv) Plotly.Plots.resize(chartDiv); } catch (e) {}\n"
        "      });\n"
        "      if (chartDiv && chartDiv.parentElement) ro.observe(chartDiv.parentElement);\n"
        "    } catch (e) {}\n"
    )

    # Final HTML payload for the embedded iframe.
    html_code = head + injected + tail
    # Provide extra room for the under-chart table and ensure the iframe
    # has an explicit height (otherwise the chart can appear blank if the iframe
    # collapses to 0px in some deployments).
    #
    # On mobile, the table can get clipped if the fixed extra margin is too small.
    # We therefore scale the extra space by the number of rows (tickers) when available.
    _extra = 240
    try:
        _n_rows = int(len((table_payload or {}).get("order") or []))
        # Approx row height ~22-26px + header; cap to avoid excessive whitespace.
        _extra = int(min(700, max(_extra, 140 + 26 * _n_rows)))
    except Exception:
        _extra = 240
    _width_arg = int(width) if (width is not None and int(width) > 0) else None
    components.html(html_code, height=int(height) + int(_extra), width=_width_arg, scrolling=False)

def render_plotly_with_hover_sync(fig, key: str, height: int = 450):
    """Render a Plotly figure.

    Note: In v5.0.4+ we render the hover-sync UI purely in the frontend via
    st.components.v1.html to avoid Streamlit custom component routing issues.
    Therefore no hovered x-value is returned to Python.
    """
    def _to_jsonable(x):
        """Convert common scientific-python objects into JSON-serializable types.

        Streamlit custom components require args to be JSON serializable.
        Plotly figures often include numpy arrays/scalars, pandas timestamps,
        and other objects that need conversion.
        """
        # Fast-path for primitives
        if x is None or isinstance(x, (str, int, float, bool)):
            return x

        # numpy arrays / pandas series -> list (then recurse)
        # tolist() may still contain non-JSON objects (e.g., datetime),
        # so we must run the result back through this converter.
        tolist = getattr(x, "tolist", None)
        if callable(tolist):
            try:
                return _to_jsonable(tolist())
            except Exception:
                pass

        # numpy scalar -> python scalar (then recurse)
        item = getattr(x, "item", None)
        if callable(item):
            try:
                return _to_jsonable(item())
            except Exception:
                pass

        # pandas Timestamp / datetime-like
        try:
            if hasattr(x, "isoformat") and callable(getattr(x, "isoformat")):
                return x.isoformat()
        except Exception:
            pass

        # dict
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}

        # list/tuple
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]

        # fallback string
        return str(x)

    try:
        fig_dict_raw = fig.to_dict()
        fig_dict = _to_jsonable(fig_dict_raw)
    except Exception:
        fig_dict = {}

    _render_plotly_hover_sync_front(fig_dict, table_payload=None, height=height, key=key, width=None)
    return None


def render_plotly_with_table_front(fig, table_payload: dict, key: str, height: int = 450, width: int | None = None):
    """Render Plotly + under-chart table entirely in the frontend.

    This avoids Streamlit reruns on high-frequency hover events.
    The component receives both the figure and the normalized series needed to
    update the table at the hovered X-position.
    """

    def _to_jsonable(x):
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        tolist = getattr(x, "tolist", None)
        if callable(tolist):
            try:
                return _to_jsonable(tolist())
            except Exception:
                pass
        item = getattr(x, "item", None)
        if callable(item):
            try:
                return _to_jsonable(item())
            except Exception:
                pass
        try:
            if hasattr(x, "isoformat") and callable(getattr(x, "isoformat")):
                return x.isoformat()
        except Exception:
            pass
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        return str(x)

    try:
        fig_dict = _to_jsonable(fig.to_dict())
    except Exception:
        fig_dict = {}
    payload = _to_jsonable(table_payload or {})
    _render_plotly_hover_sync_front(fig_dict, table_payload=payload, height=height, key=key, width=width)
    return None


from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages


def _render_change_table_under_chart(
    label: str,
    norm_df: pd.DataFrame,
    name_map: dict[str, str] | None = None,
    max_rows: int | None = None,
    hover_x: object | None = None,
    color_map: dict[str, str] | None = None,
):
    """Render a % change table directly under a normalized chart.

    - norm_df values are normalized levels (Start=1.0).
    - If hover_x is provided, the table uses the last available row at or before hover_x.
      Otherwise, it uses the latest row.
    - If color_map is provided, show a small color swatch in the leftmost column.

    This is used as a "legend under the chart".
    """
    if norm_df is None or norm_df.empty:
        return

    df = norm_df.ffill()

    ref_ts = None
    if hover_x is not None:
        try:
            ref_ts = pd.to_datetime(hover_x)
        except Exception:
            ref_ts = None

    try:
        if ref_ts is not None:
            df2 = df.loc[:ref_ts]
            if df2.empty:
                row = df.iloc[0]
                ref_ts = df.index[0]
            else:
                row = df2.iloc[-1]
                ref_ts = df2.index[-1]
        else:
            row = df.iloc[-1]
            ref_ts = df.index[-1]
    except Exception:
        return

    rows: list[tuple[str, str, float]] = []
    for key, v in row.items():
        if pd.isna(v):
            continue
        key_s = str(key)
        nm = name_map.get(key, key) if name_map else key_s
        try:
            chg = (float(v) - 1.0) * 100.0
        except Exception:
            continue
        rows.append((key_s, str(nm), chg))

    if not rows:
        return

    # Sort by magnitude (most informative)
    rows.sort(key=lambda t: abs(t[2]), reverse=True)

    if max_rows is not None:
        rows = rows[:max_rows]

    ref_label = ref_ts.strftime('%Y-%m-%d') if hasattr(ref_ts, 'strftime') else str(ref_ts)

    html = [
        f"<div style='margin-top:0.25rem; margin-bottom:0.25rem; font-weight:600;'>{label}（{ref_label} / Start=1基準）</div>",
        "<div style='overflow-x:auto'>",
        "<table style='border-collapse:collapse; width:100%; font-size:0.90rem;'>",
        "<thead><tr>",
        "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #ddd; width:28px;'></th>",
        "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #ddd;'>銘柄</th>",
        "<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #ddd;'>Ticker</th>",
        "<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #ddd;'>変化率(%)</th>",
        "</tr></thead><tbody>",
    ]

    for key_s, nm, chg in rows:
        color = color_map.get(key_s) if color_map else None
        swatch = ""
        if color:
            swatch = (
                f"<span style='display:inline-block; width:14px; height:14px; "
                f"border-radius:3px; background:{color};'></span>"
            )
        chg_str = f"{chg:+.2f}"
        html.append(
            "<tr>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #f0f0f0;'>{swatch}</td>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #f0f0f0;'>{nm}</td>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #f0f0f0;'>{key_s}</td>"
            f"<td style='padding:6px 8px; border-bottom:1px solid #f0f0f0; text-align:right;'>{chg_str}</td>"
            "</tr>"
        )

    html.extend(["</tbody></table></div>"])
    st.markdown("\n".join(html), unsafe_allow_html=True)


# 任意：日本語文字化け対策（入っていれば）
try:
    import japanize_matplotlib  # noqa
except Exception:
    pass


# =========================
# Modularized imports
# =========================
from asset_app.config.constants import TAX_RATE_DEFAULT, REP_MARKET_UNIVERSE
from asset_app.config.version import VERSION
from asset_app.infra.export import (
    fig_to_png_bytes,
    df_to_csv_bytes,
    series_to_csv_bytes,
    add_df_as_table_page,
    build_export_pdf_bytes,
)
from asset_app.infra.instruments import (
    load_instruments,
    yfinance_symbol_search,
    YahooRateLimitError,
    load_us_tickers,
    search_us_tickers_local,
)
from asset_app.infra.jpx_local import (
    JPX_CACHE_PATH,
    JPX_SOURCE_CANDIDATES,
    get_jpx_source_path,
    load_jpx_source,
    search_jpx_candidates,
)
from asset_app.infra.market_data import (
    guess_asset_type_from_quote_type,
    fetch_adjclose,
    fetch_usdjpy,
    fetch_market_caps,
    get_rep_market_prices_jpy,
)
from asset_app.domain.fx import to_jpy, align_common_start
from asset_app.domain.portfolio import (
    build_buy_and_hold_pf,
    calc_metrics_from_pf,
    get_rebalance_dates,
    build_rebalanced_pf_with_tax,
    build_threshold_rebalanced_pf_with_tax,
    build_walk_forward_pf_with_tax,
)
from asset_app.services.black_litterman import (
    read_bl_settings,
    compute_bl_posterior_returns,
    _HAVE_PYPORTFOLIOOPT as _HAVE_PYPORTFOLIOOPT_BL,
)
from asset_app.services.optimizer import (
    _HAVE_PYPORTFOLIOOPT,
    _HAVE_SCIPY,
    optimize_weights_pyportfolioopt,
    optimize_weights_scipy,
    optimize_weights,
    optimize_weights_cached,
)
from asset_app.infra.portfolio_store import (
    load_portfolios,
    save_portfolios,
    portfolio_to_df_items,
)
from asset_app.config.paths import (
    default_portfolios_path,
    default_instruments_path,
    default_us_tickers_path,
    default_us_mcap_cache_path,
    default_us_meta_cache_path,
)
from asset_app.state.init import ensure_state_defaults
from asset_app.ui.portfolio_widgets import (
    compute_weights_from_weight_widgets,
    compute_weights_from_widgets,
    get_current_pf_state,
    set_current_pf_state,
    apply_weights_to_weight_widgets,
)

# Note: BL helper module has its own _HAVE_PYPORTFOLIOOPT flag;
# we keep the UI behavior consistent by using the optimizer flag for availability.


# =========================
# UI
# =========================
# URL例:
#   通常:        http://<host>:8501/
#   閲覧専用:    http://<host>:8501/?view=readonly
qp = st.query_params
_view = qp.get("view", "")
if isinstance(_view, (list, tuple)):
    _view = _view[0] if _view else ""
readonly = str(_view).lower() == "readonly"

# Ensure session_state has all required defaults before any UI logic reads it.
ensure_state_defaults(st.session_state)

# ------------------------------------------------------------
# Quick date-range: apply pending changes BEFORE sidebar widgets
# ------------------------------------------------------------
def _apply_pending_date_range() -> None:
    """Apply date range requested by quick-range buttons.

    Streamlit forbids mutating st.session_state for a widget key after the widget
    is created. We therefore store a *pending* request in session_state, then apply
    it at the top of the script before the sidebar date widgets are instantiated.
    """
    req = st.session_state.pop("_quick_range_pending", None)
    if not isinstance(req, dict):
        return
    start = req.get("start")
    end = req.get("end")
    if start:
        st.session_state["start_date"] = str(start)
    if end:
        st.session_state["end_date"] = str(end)


_apply_pending_date_range()

st.set_page_config(
    page_title="PF Checker",
    layout="wide",
    initial_sidebar_state="collapsed" if readonly else "expanded",
)
st.title("PF Checker")

# Build/version stamp (helps verify which deployment is running)
st.caption(f"Version {VERSION}")

if readonly:
    st.info("閲覧専用モードです（URL: ?view=readonly）。設定変更はできません。")

st.sidebar.header("設定")
st.sidebar.caption(f"Version {VERSION}")

# 入力変更のたびに全再計算されるのを防ぐ（スマホ/低スペック環境向け）
run = True
if not readonly:
    run = st.sidebar.button("更新")

if readonly:
    csv_path = str(default_instruments_path())
    df_inst = load_instruments(csv_path)
    names = df_inst["display_name"].tolist()
    # （閲覧専用では instruments.csv の編集UIは非表示）
    st.markdown('---')
    st.write('現在の instruments.csv（先頭30行）')
    st.dataframe(df_inst.head(30), use_container_width=True)

    selected_names = names[:3]
    benchmark_name = "なし"
    start_date = "2024-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    initial_value = 1_000_000
    rf_annual = 0.04
    trading_days_per_year = 245.0

    rebalance_mode = "なし（買いっぱなし）"
    tax_rate = TAX_RATE_DEFAULT
    rebalance_rule = "カレンダー（D/W/M/Y）"
    threshold_rel = 0.10
    cooldown_days = 0
    min_target_for_rel = 0.01

    fig_width = 12
    fig_height = 6
    fig_dpi = 100

    try:
        _sd = pd.to_datetime(start_date)
    except Exception:
        _sd = pd.Timestamp("2024-01-01")

    default_train_end = (_sd - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    default_train_start = (_sd - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    train_start_date = default_train_start
    train_end_date = default_train_end

    opt_choices = ["MaxSharpe"]
else:
    # =========================
    # ポートフォリオ管理（portfolios.json）
    #   - instruments.csv は代表ETF市場の内部用途として保持（UIから編集しない）
    #   - ユーザーの銘柄選択・ウェイトは portfolios.json に保存/読込
    # =========================
    PORTFOLIOS_PATH = str(default_portfolios_path())


    pf_store = load_portfolios(PORTFOLIOS_PATH)
    pf_ids = sorted(pf_store.get("portfolios", {}).keys())

    with st.sidebar.expander("ポートフォリオ（保存/読込）", expanded=True):
        st.caption("銘柄選択とウェイトは portfolios.json に保存します（推奨配置: /home/pi/portfolios.json）。instruments.csv は編集しません（代表ETF市場の内部用途のみ）。")

        sel_pf_id = st.selectbox("保存済みポートフォリオ", options=["(未選択)"] + pf_ids, index=0)
        col_a, col_b = st.columns(2)
        with col_a:
            load_clicked = st.button("読み込み", use_container_width=True)
        with col_b:
            clear_clicked = st.button("クリア", use_container_width=True)

        if clear_clicked:
            set_current_pf_state([], {})
            st.session_state["_overwrite_armed"] = None
            st.success("現在のポートフォリオをクリアしました。")
            st.rerun()

        if load_clicked and sel_pf_id != "(未選択)":
            pf = pf_store["portfolios"].get(sel_pf_id, {})
            items = pf.get("items", [])
            weights = pf.get("weights", {})
            if not isinstance(items, list):
                items = []
            if not isinstance(weights, dict):
                weights = {}
            set_current_pf_state(items, weights)
            # Reflect saved weights into weight widgets so UI matches portfolios.json
            apply_weights_to_weight_widgets(items, weights)
            st.session_state["_overwrite_armed"] = None
            st.success(f"読み込みました: {sel_pf_id}")
            st.rerun()

        st.markdown("##### 保存")
        new_pf_id = st.text_input("ID（英数字/アンダースコア推奨）", key="pf_new_id")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            save_new = st.button("保存（新規）", use_container_width=True)
        with col_s2:
            arm_overwrite = st.button("上書き（準備）", use_container_width=True)

        items, weights_by_ticker = get_current_pf_state()
        # validate current
        df_pf_tmp = portfolio_to_df_items(items)
        if df_pf_tmp.empty:
            st.info("現在のポートフォリオに銘柄がありません。銘柄管理から追加してください。")

        if save_new:
            if not new_pf_id.strip():
                st.error("IDを入力してください。")
            elif new_pf_id in pf_store["portfolios"]:
                st.error("そのIDは既に存在します。上書きを使ってください。")
            else:
                w = compute_weights_from_widgets(items)
                if not w and not df_pf_tmp.empty:
                    # fallback
                    tickers = df_pf_tmp["ticker"].tolist()
                    eq = 1.0 / len(tickers)
                    w = {t: eq for t in tickers}
                pf_store["portfolios"][new_pf_id] = {
                    "name": new_pf_id,
                    "created_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
                    "items": df_pf_tmp.to_dict(orient="records"),
                    "weights": w,
                    "notes": "",
                }
                save_portfolios(PORTFOLIOS_PATH, pf_store)
                st.session_state["_overwrite_armed"] = None
                st.success(f"保存しました: {new_pf_id}")
                st.rerun()

        # two-step overwrite
        active_id = sel_pf_id if sel_pf_id != "(未選択)" else ""
        if arm_overwrite:
            if not active_id:
                st.error("上書き対象を「保存済みポートフォリオ」から選んでください。")
            else:
                st.session_state["_overwrite_armed"] = active_id
                st.warning(f"上書き準備完了: {active_id}（次のボタンで確定）")

        if st.session_state.get("_overwrite_armed"):
            confirm_overwrite = st.button("上書きを確定（取り消し不可）", type="primary", use_container_width=True)
            cancel_overwrite = st.button("上書きをキャンセル", use_container_width=True)
            if cancel_overwrite:
                st.session_state["_overwrite_armed"] = None
                st.info("キャンセルしました。")
                st.rerun()
            if confirm_overwrite:
                target = st.session_state.get("_overwrite_armed")
                if not target or target not in pf_store["portfolios"]:
                    st.error("上書き対象が見つかりません。")
                else:
                    w = compute_weights_from_widgets(items)
                    if not w and not df_pf_tmp.empty:
                        tickers = df_pf_tmp["ticker"].tolist()
                        eq = 1.0 / len(tickers)
                        w = {t: eq for t in tickers}
                    pf_store["portfolios"][target]["items"] = df_pf_tmp.to_dict(orient="records")
                    pf_store["portfolios"][target]["weights"] = w
                    pf_store["portfolios"][target]["name"] = pf_store["portfolios"][target].get("name") or target
                    save_portfolios(PORTFOLIOS_PATH, pf_store)
                    st.session_state["_overwrite_armed"] = None
                    st.success(f"上書きしました: {target}")
                    st.rerun()
        st.markdown("##### 削除")
        # 削除対象は『保存済みポートフォリオ』の選択とは独立させる
        pf_ids_for_delete = sorted(list(pf_store.get("portfolios", {}).keys()))

        # Streamlitの制約: ウィジェット生成後に同じkeyのsession_stateを変更できない。
        # 削除後の「(未選択)」リセットは、次回実行の冒頭で行う。
        if st.session_state.pop("_pf_delete_reset", False):
            st.session_state["pf_delete_target_id"] = "(未選択)"

        del_pf_id = st.selectbox("削除するポートフォリオID", options=["(未選択)"] + pf_ids_for_delete, index=0, key="pf_delete_target_id")
        del_active_id = del_pf_id if del_pf_id != "(未選択)" else ""

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            arm_delete = st.button("削除（準備）", use_container_width=True)
        with col_d2:
            st.caption("※ 二段階確認")

        if arm_delete:
            if not del_active_id:
                st.error("削除対象を選んでください（この欄は『保存済みポートフォリオ』とは別です）。")
            else:
                st.session_state["_delete_armed"] = del_active_id
                st.warning(f"削除準備完了: {del_active_id}（次のボタンで確定）")

        if st.session_state.get("_delete_armed"):
            confirm_delete = st.button("削除を確定（取り消し不可）", type="primary", use_container_width=True)
            cancel_delete = st.button("削除をキャンセル", use_container_width=True)
            if cancel_delete:
                st.session_state["_delete_armed"] = None
                st.info("キャンセルしました。")
                st.rerun()
            if confirm_delete:
                target_del = st.session_state.get("_delete_armed")
                if not target_del or target_del not in pf_store.get("portfolios", {}):
                    st.error("削除対象が見つかりません。")
                else:
                    del pf_store["portfolios"][target_del]
                    save_portfolios(PORTFOLIOS_PATH, pf_store)
                    st.session_state["_delete_armed"] = None
                    st.session_state["_overwrite_armed"] = None
                    # 削除後は削除対象選択をリセット
                    st.session_state["_pf_delete_reset"] = True
                    st.success(f"削除しました: {target_del}")
                    st.rerun()

    # =========================
    # 銘柄管理（検索→自動入力→ポートフォリオに追加）
    # =========================
    with st.sidebar.expander("銘柄管理（検索→自動入力→ポートフォリオへ追加）", expanded=False):
        st.caption("銘柄名/ティッカーを検索して候補から選ぶと、ティッカー等を自動入力し、現在のポートフォリオに追加できます。instruments.csv には書き込みません。")

        st.markdown("##### 検索")

        # JPX 上場銘柄一覧（data_j.xls）による漢字検索
        jpx_src = get_jpx_source_path()
        if jpx_src is None:
            st.caption("JPX銘柄一覧: data_j.xls が見つからないため、漢字検索は無効です（/home/pi/data_j.xls を配置してください）。")
            df_jpx = pd.DataFrame()
        else:
            try:
                df_jpx = load_jpx_source(jpx_src)
            except Exception as e:
                st.warning(f"JPX銘柄一覧の読み込みに失敗しました: {e}")
                df_jpx = pd.DataFrame()

        # --- Search is explicitly triggered to avoid Yahoo rate limiting (HTTP 429) ---
        if "pf_add_search_last_query" not in st.session_state:
            st.session_state["pf_add_search_last_query"] = ""
        if "pf_add_search_results" not in st.session_state:
            st.session_state["pf_add_search_results"] = {"jpx": [], "us_local": [], "yf": [], "yf_429": False}

        with st.form("pf_add_search_form", clear_on_submit=False):
            q_input = st.text_input("検索キーワード（漢字/カナ/英語/ティッカー）", key="pf_add_query_input")
            submitted = st.form_submit_button("検索")

        if submitted:
            q = (q_input or "").strip()
            st.session_state["pf_add_search_last_query"] = q

            # 1) JPXローカル候補（日本株）
            jpx_candidates = []
            if q and not df_jpx.empty:
                jpx_candidates = search_jpx_candidates(df_jpx, q, limit=30)

            # 2) ローカル米国ティッカー候補（任意: us_tickers.csv）
            us_local_candidates = []
            us_path = default_us_tickers_path()
            if q and us_path.exists():
                try:
                    df_us = load_us_tickers(str(us_path))
                    mcap_path = default_us_mcap_cache_path()
                    meta_path = default_us_meta_cache_path()
                    us_local_candidates = search_us_tickers_local(
                        df_us,
                        q,
                        limit=30,
                        mcap_cache_path=str(mcap_path) if mcap_path.exists() else None,
                        meta_cache_path=str(meta_path) if meta_path.exists() else None,
                    )
                except Exception as e:
                    # Do not silently drop results; make diagnostics available.
                    st.session_state["pf_add_us_local_error"] = str(e)
                    us_local_candidates = []
            else:
                st.session_state.pop("pf_add_us_local_error", None)

            # 3) Yahoo Finance候補（海外含む / レート制限対策済み）
            yf_candidates = []
            yf_429 = False
            # Cooldown for Yahoo Finance symbol search (HTTP 429). When Yahoo blocks our IP,
            # repeatedly retrying only prolongs the block. We persist a short "block until"
            # timestamp under ASSET_APP_DATA_DIR so it survives service restarts.
            try:
                import time as _time
                import json as _json

                _block_path = (DATA_DIR / "yf_symbol_search_block.json")
                _now = int(_time.time())
                _block_until = 0
                if "yf_symbol_search_block_until" in st.session_state:
                    _block_until = int(st.session_state.get("yf_symbol_search_block_until") or 0)
                elif _block_path.exists():
                    try:
                        _obj = _json.loads(_block_path.read_text(encoding="utf-8"))
                        _block_until = int(_obj.get("block_until", 0))
                    except Exception:
                        _block_until = 0
                if _block_until and _now < _block_until:
                    yf_429 = True
            except Exception:
                pass
            # IMPORTANT: Do NOT call Yahoo search when local dictionaries already returned results.
            # This avoids annoying HTTP 429 and reduces network dependency.
            local_hit = bool(jpx_candidates) or bool(us_local_candidates)
            if q and not local_hit and not yf_429:
                try:
                    yf_candidates = yfinance_symbol_search(q, max_results=30)
                except YahooRateLimitError:
                    yf_429 = True
                    yf_candidates = []
                    # Persist a cooldown to avoid hammering Yahoo (default: 24 hours).
                    try:
                        import time as _time
                        import json as _json

                        _now = int(_time.time())
                        _block_until = _now + 24 * 3600
                        st.session_state["yf_symbol_search_block_until"] = _block_until
                        (DATA_DIR / "yf_symbol_search_block.json").write_text(
                            _json.dumps({"block_until": _block_until, "reason": "http_429"}),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                except Exception:
                    yf_candidates = []

            st.session_state["pf_add_search_results"] = {
                "jpx": jpx_candidates,
                "us_local": us_local_candidates,
                "yf": yf_candidates,
                "yf_429": yf_429,
            }

        q = (st.session_state.get("pf_add_search_last_query") or "").strip()
        jpx_candidates = st.session_state.get("pf_add_search_results", {}).get("jpx", []) or []
        us_local_candidates = st.session_state.get("pf_add_search_results", {}).get("us_local", []) or []
        yf_candidates = st.session_state.get("pf_add_search_results", {}).get("yf", []) or []
        yf_429 = bool(st.session_state.get("pf_add_search_results", {}).get("yf_429", False))

        us_local_err = (st.session_state.get("pf_add_us_local_error") or "").strip()
        if us_local_err:
            st.warning(f"ローカル米国ティッカー検索でエラーが発生しました: {us_local_err}")

        if yf_429:
            st.warning(
                "Yahoo Finance 検索がレート制限（HTTP 429）でブロックされています。"
                "しばらく待つか、ティッカーを直接入力してください。"
                "（候補表示はローカル辞書 / us_tickers.csv の範囲で継続します。）"
            )

        if not default_us_tickers_path().exists():
            st.caption(
                "米国株/暗号資産の候補表示を安定させるには、外部データディレクトリに us_tickers.csv（symbol,name,exchange[,asset_class]）を配置してください。"
            )

        # 統合候補リスト
        options = []
        cand_map = {}
        for c in jpx_candidates:
            label = f"JPX: {c['display_name']} ({c['ticker']})"
            options.append(label)
            cand_map[label] = c
        for c in us_local_candidates:
            label = f"US(local): {c.get('name','') or c.get('symbol','')} ({c.get('symbol','')})"
            options.append(label)
            cand_map[label] = c
        for c in yf_candidates:
            label = f"YF: {c['name']} ({c['symbol']})"
            options.append(label)
            cand_map[label] = c

        chosen_label = st.selectbox("候補", options=["(未選択)"] + options, index=0)
        if chosen_label != "(未選択)":
            chosen = cand_map[chosen_label]
            if chosen_label.startswith("JPX:"):
                st.session_state["new_display_name"] = chosen.get("display_name", "")
                st.session_state["new_ticker"] = chosen.get("ticker", "")
                st.session_state["new_asset_type"] = "equity"
                st.session_state["pf_new_asset_type_ui"] = "equity"
                st.session_state["new_currency"] = "JPY"
                st.session_state["pf_new_currency_ui"] = "JPY"
                st.session_state["pf_new_currency_ui"] = "JPY"
                st.session_state["pf_new_asset_type_ui"] = "equity"
            else:
                st.session_state["new_display_name"] = chosen.get("name", "") or chosen.get("symbol", "")
                st.session_state["new_ticker"] = chosen.get("symbol", "")
                st.session_state["new_asset_type"] = guess_asset_type_from_quote_type(chosen.get("quoteType", ""))
                # keep UI selectboxes in sync (selectbox uses its own key state once created)
                st.session_state["pf_new_asset_type_ui"] = st.session_state.get("new_asset_type", "equity")
                cur = (chosen.get("currency") or "").strip().upper()
                sym = (chosen.get("symbol") or "").strip()
                if not cur:
                    cur = "JPY" if sym.upper().endswith(".T") else "USD"
                st.session_state["new_currency"] = cur
                st.session_state["pf_new_currency_ui"] = cur
                st.session_state["pf_new_currency_ui"] = cur
                st.session_state["pf_new_currency_ui"] = cur

        st.markdown("##### 追加（現在のポートフォリオ）")
        st.text_input("表示名", key="new_display_name")
        st.text_input("Ticker", key="new_ticker")

        asset_type_options = ["equity", "etf", "fund", "bond", "crypto", "cash", "other"]
        currency_options = ["JPY", "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "CNY", "HKD", "SGD", "KRW", "other"]

        at_default = st.session_state.get("new_asset_type") or "equity"
        cc_default = st.session_state.get("new_currency") or "JPY"

        new_asset_type = st.selectbox("資産タイプ", options=asset_type_options, index=asset_type_options.index(at_default) if at_default in asset_type_options else 0, key="pf_new_asset_type_ui")
        new_currency = st.selectbox("通貨", options=currency_options, index=currency_options.index(cc_default) if cc_default in currency_options else 0, key="pf_new_currency_ui")

        col_add1, col_add2 = st.columns(2)
        with col_add1:
            add_to_pf = st.button("追加", use_container_width=True)
        with col_add2:
            upsert_pf = st.button("上書き（同Ticker置換）", use_container_width=True)

        if add_to_pf or upsert_pf:
            items, weights_by_ticker = get_current_pf_state()
            df_pf = portfolio_to_df_items(items)
            t = (st.session_state.get("new_ticker") or "").strip()
            dn = (st.session_state.get("new_display_name") or "").strip() or t
            if not t:
                st.error("Tickerを入力してください。")
            else:
                row = {"display_name": dn, "ticker": t, "asset_type": new_asset_type, "currency": new_currency}
                # upsert by ticker
                existing = [x for x in items if str(x.get("ticker", "")).strip() == t]
                if existing and not upsert_pf:
                    st.error("同Tickerが既に存在します。上書きを使ってください。")
                else:
                    # replace or append
                    new_items = [x for x in items if str(x.get("ticker", "")).strip() != t]
                    new_items.append(row)
                    # keep weights; if new ticker, set default 1/N (do not renormalize existing)
                    if t not in weights_by_ticker:
                        n = len(new_items)
                        weights_by_ticker[t] = 1.0 / max(1, n)
                    # ensure weight widget key exists for new/updated display_name
                    try:
                        st.session_state[f"w_{dn}"] = float(weights_by_ticker.get(t, 0.0) or 0.0)
                    except Exception:
                        pass
                    set_current_pf_state(new_items, weights_by_ticker)
                    st.success("ポートフォリオに反映しました。")
                    st.rerun()

        st.markdown("---")
        items, weights_by_ticker = get_current_pf_state()
        df_pf_preview = portfolio_to_df_items(items)
        st.write("現在のポートフォリオ（銘柄一覧）")
        st.dataframe(df_pf_preview, use_container_width=True)

        # Remove items from current portfolio
        if not df_pf_preview.empty:
            st.markdown("##### 削除（現在のポートフォリオ）")
            _opts = [f"{r['display_name']} ({r['ticker']})" for _, r in df_pf_preview.iterrows()]
            _sel = st.multiselect("削除する銘柄", options=_opts, default=[])
            if st.button("選択した銘柄を削除", use_container_width=True):
                to_remove = set()
                for s in _sel:
                    # extract ticker in parentheses
                    if '(' in s and s.endswith(')'):
                        to_remove.add(s.split('(')[-1][:-1])
                items, weights_by_ticker = get_current_pf_state()
                new_items = [x for x in items if str(x.get('ticker','')).strip() not in to_remove]
                for t in list(weights_by_ticker.keys()):
                    if t in to_remove:
                        weights_by_ticker.pop(t, None)
                set_current_pf_state(new_items, weights_by_ticker)
                # also clean weight widget keys to avoid stale UI
                for x in items:
                    dn = str(x.get('display_name','')).strip()
                    if str(x.get('ticker','')).strip() in to_remove and dn:
                        st.session_state.pop(f"w_{dn}", None)
                st.success("削除しました。")
                st.rerun()

    # 現在のポートフォリオを以後の計算で使用
    items, weights_by_ticker = get_current_pf_state()
    df_inst = portfolio_to_df_items(items)
    names = df_inst["display_name"].tolist()

    # ベンチマーク候補: 現在PF + instruments.csv（代表ETF市場の内部銘柄も候補に含める）
    try:
        df_inst_fixed = load_instruments(str(default_instruments_path()))
    except Exception:
        df_inst_fixed = pd.DataFrame(columns=["display_name", "ticker", "asset_type", "currency"])

    # ベンチマーク検索用の統合テーブル（同名はPF側を優先）
    try:
        df_bench_all = pd.concat([df_inst, df_inst_fixed], ignore_index=True)
        df_bench_all = df_bench_all.drop_duplicates(subset=["display_name"], keep="first")
    except Exception:
        df_bench_all = df_inst.copy()

    bench_names = df_bench_all["display_name"].astype(str).tolist()

    # 表示名→ticker のマップ（weights保存用）
    _dn_to_tk = df_inst.set_index("display_name")["ticker"].astype(str).to_dict()
    # ticker→display_name のマップ
    _tk_to_dn = {v: k for k, v in _dn_to_tk.items()}

    # デフォルト選択：ポートフォリオに入っている銘柄（なければ空）
    selected_names = names.copy()
# --- PF選択銘柄のticker候補を保存（相対ビュー等で使用） ---
try:
    _name_to_ticker = df_inst.set_index("display_name")["ticker"].astype(str).to_dict()
    _pf_view_assets = [str(_name_to_ticker[n]) for n in selected_names if n in _name_to_ticker]
    st.session_state["_pf_view_assets"] = list(dict.fromkeys(_pf_view_assets))
except Exception:
    st.session_state["_pf_view_assets"] = []


st.sidebar.markdown("---")
benchmark_name = st.sidebar.selectbox(
    "ベンチマーク（比較対象）", options=["なし"] + bench_names, index=0
)

start_date = st.sidebar.text_input(
    "開始日 (YYYY-MM-DD)",
    value=str(st.session_state.get("start_date", "2024-01-01")),
    key="start_date",
)
end_date = st.sidebar.text_input(
    "終了日 (YYYY-MM-DD)",
    value=str(st.session_state.get("end_date", pd.Timestamp.today().strftime("%Y-%m-%d"))),
    key="end_date",
)

initial_value = st.sidebar.number_input(
    "初期投資額 (JPY)", min_value=1_000, value=1_000_000, step=100_000
)

rf_annual = st.sidebar.number_input(
    "rf 年率（例: 0.04 = 4%）", min_value=0.0, value=0.04, step=0.01, format="%.4f"
)
trading_days_per_year = st.sidebar.number_input(
    "東証営業日数/年（推定）", min_value=200.0, max_value=260.0, value=245.0, step=1.0
)

# ----- 固定ウェイト：リバランス -----
st.sidebar.markdown("---")
st.sidebar.subheader("固定ウェイト：リバランス")

rebalance_mode = st.sidebar.selectbox(
    "頻度（カレンダー）",
    options=["なし（買いっぱなし）", "日次(D)", "週次(W)", "月次(M)", "年次(Y)"],
    index=0,
)
tax_rate = st.sidebar.number_input(
    "売却益税率",
    min_value=0.0,
    max_value=1.0,
    value=TAX_RATE_DEFAULT,
    step=0.00001,
    format="%.5f",
)

rebalance_rule = st.sidebar.selectbox(
    "実行ルール",
    options=["カレンダー（D/W/M/Y）", "相対ズレしきい値"],
    index=0,
)

threshold_rel = st.sidebar.number_input(
    "しきい値（相対ズレ）例: 0.10 = 目標比率から±10%ズレたら実行",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05,
    format="%.3f",
)

cooldown_days = st.sidebar.number_input(
    "最短間隔（営業日）0で無効",
    min_value=0,
    max_value=260,
    value=20,
    step=1,
)

min_target_for_rel = st.sidebar.number_input(
    "相対ズレの分母下限（目標ウェイト下限）例: 0.01 = 1%",
    min_value=0.0001,
    max_value=0.20,
    value=0.01,
    step=0.001,
    format="%.4f",
)

# ----- グラフサイズ -----
st.sidebar.markdown("---")
st.sidebar.subheader("グラフサイズ（表示）")
display_width_px = st.sidebar.slider(
    "横幅（px）", min_value=360, max_value=1600, value=1200, step=20
)
display_height_px = st.sidebar.slider(
    "縦幅（px）", min_value=200, max_value=1000, value=320, step=20
)

# Display size is controlled in px for easier tuning across devices.
# Matplotlib uses inch x DPI, so we convert with a fixed display DPI.
_DISPLAY_DPI = 100
fig_width = float(display_width_px) / float(_DISPLAY_DPI)
fig_height = float(display_height_px) / float(_DISPLAY_DPI)
fig_dpi = int(_DISPLAY_DPI)

# Plotly height (px). Streamlit's Plotly renderer respects layout.height.
plotly_height_px = int(max(200, display_height_px))

# PDF export sizing (legacy inch/DPI controls)
with st.sidebar.expander("PDF出力サイズ（inch/DPI）", expanded=False):
    pdf_fig_width = st.slider("横幅（inch）", min_value=6, max_value=20, value=12, step=1, key="pdf_fig_width_in")
    pdf_fig_height = st.slider("縦幅（inch）", min_value=3, max_value=14, value=4, step=1, key="pdf_fig_height_in")
    pdf_fig_dpi = st.slider("DPI", min_value=80, max_value=200, value=100, step=10, key="pdf_fig_dpi")

# ----- 静的最適化：学習期間 -----
st.sidebar.markdown("---")
st.sidebar.subheader("最適化（静的）：学習期間（開始日前）")

try:
    _sd = pd.to_datetime(start_date)
except Exception:
    _sd = pd.Timestamp("2024-01-01")

default_train_end = (_sd - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
default_train_start = (_sd - pd.DateOffset(years=3)).strftime("%Y-%m-%d")

train_start_date = st.sidebar.text_input(
    "学習開始日 (YYYY-MM-DD)", value=default_train_start
)
train_end_date = st.sidebar.text_input(
    "学習終了日 (YYYY-MM-DD)", value=default_train_end
)

_OPT_METHOD_OPTIONS = ["MaxSharpe", "MinVol", "BL_MaxSharpe", "BL_MinVol"]

# NOTE:
# Streamlit は任意の操作でスクリプト全体が再実行されます。
# その際、widget の状態が何らかの理由で空配列になったり、別処理の rerun で
# 初期値が再適用されて選択が外れると、UI上でも BL_MaxSharpe が「未選択」に見え、
# 対応する系列が表示対象から外れてしまいます。
#
# ここでは widget state（opt_choices_ui）とは別に、最後にユーザーが確定した選択を
# opt_choices_persist に保持し、再実行で opt_choices_ui が空に戻った場合に復元します。
if "opt_choices_persist" not in st.session_state:
    st.session_state["opt_choices_persist"] = ["MaxSharpe"]


def _on_opt_choices_change() -> None:
    """Persist optimization method selections only when the user changes the widget.

    Streamlit reruns frequently on unrelated interactions (e.g., adding/removing tickers).
    In those reruns, widget state can transiently fall back to defaults. If we blindly
    overwrite persist on every rerun, BL_MaxSharpe can be dropped unintentionally.
    """

    cur = st.session_state.get("opt_choices_ui")
    if isinstance(cur, list):
        st.session_state["opt_choices_persist"] = [x for x in cur if x in _OPT_METHOD_OPTIONS]


# --- Robust restore ---
# Always seed the widget key from the persisted selection BEFORE creating the widget.
_pv = st.session_state.get("opt_choices_persist") or ["MaxSharpe"]
st.session_state["opt_choices_ui"] = [x for x in _pv if x in _OPT_METHOD_OPTIONS]

opt_choices = st.sidebar.multiselect(
    "追加する最適化ポートフォリオ（評価期間はBuy&Hold）",
    options=_OPT_METHOD_OPTIONS,
    key="opt_choices_ui",
    on_change=_on_opt_choices_change,
)

# 明示的に「空」にしたい場合のみクリアを許可する（空の確定はボタンのみ）。
if st.sidebar.button(
    "最適化ポートフォリオ選択をクリア",
    key="btn_opt_choices_clear",
    help="最適化ポートフォリオの選択状態を明示的にリセットします。",
):
    st.session_state["opt_choices_persist"] = []
    st.session_state["opt_choices_ui"] = []
    opt_choices = []



# ---- Persist BL widget states across unrelated reruns (e.g., adding instruments) ----
# Streamlit may rerun the script and, in edge cases, widget state can fall back to defaults.
# We keep a separate persisted copy and restore *before* widget instantiation.
if "bl_settings_persist" not in st.session_state:
    st.session_state["bl_settings_persist"] = {
        "bl_market_model_ui": st.session_state.get("bl_market_model_ui", st.session_state.get("_bl_market_model", "PF時価総額/AUM（簡易）")),
        "bl_risk_profile_ui": st.session_state.get("bl_risk_profile_ui", None),
        "bl_alpha_ui": st.session_state.get("bl_alpha_ui", float(st.session_state.get("_bl_alpha", 1.0))),
        "bl_use_views_ui": bool(st.session_state.get("bl_use_views_ui", False)),
    }

# Restore missing widget keys from persisted values (must happen BEFORE widgets are created)
_p = st.session_state.get("bl_settings_persist") or {}
for _k, _v in _p.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ----- Black-Litterman Views / Blend (静的最適化 & WF 共通) -----
with st.sidebar.expander("BL設定（Views / ブレンド）", expanded=False):
    # 市場モデル
    bl_market_model_options = ["PF時価総額/AUM（簡易）", "代表ETF市場（資産クラス）"]
    _bl_mm_default = str(st.session_state.get("bl_settings_persist", {}).get("bl_market_model_ui") or st.session_state.get("bl_market_model_ui") or st.session_state.get("_bl_market_model", "PF時価総額/AUM（簡易）"))
    if _bl_mm_default not in bl_market_model_options:
        _bl_mm_default = "PF時価総額/AUM（簡易）"
    bl_market_model = st.selectbox(
    "Black-Litterman：市場モデル",
    options=bl_market_model_options,
    index=bl_market_model_options.index(_bl_mm_default),
    key="bl_market_model_ui",
    )
    # persist
    st.session_state["_bl_market_model"] = bl_market_model
    try:
        st.session_state["bl_settings_persist"]["bl_market_model_ui"] = bl_market_model
    except Exception:
        pass


    # リスク回避度δは数値を直接触らせず、「リスク許容度」として抽象化して扱う
    st.markdown("#### リスク許容度（BLのリスク回避度 δ に変換）")
    _risk_labels = ["保守（リスク低）", "標準", "積極（リスク高）"]
    _risk_to_delta = {
        "保守（リスク低）": 6.0,
        "標準": 3.0,
        "積極（リスク高）": 1.5,
    }
    # 既存の_deltaがあれば近いラベルに丸める
    _delta_current = float(st.session_state.get("_bl_delta", 2.5))
    if _delta_current >= 5.0:
        _risk_default = "保守（リスク低）"
    elif _delta_current <= 2.0:
        _risk_default = "積極（リスク高）"
    else:
        _risk_default = "標準"

    risk_profile = st.selectbox(
        "リスク許容度",
        options=_risk_labels,
        index=_risk_labels.index(_risk_default),
        key="bl_risk_profile_ui",
        help="内部ではBlack-Littermanのリスク回避度δに変換して使用します。保守ほどδが大きくなり、リスクを取りにくい配分になります。",
    )
    # Robust: risk_profile can be None on rerun; default to '標準'
    rp_key = '標準' if risk_profile is None else str(risk_profile)
    delta = float(_risk_to_delta.get(rp_key, _risk_to_delta.get('標準', 3.0)))
    st.session_state["_bl_delta"] = float(delta)
    try:
        st.session_state["bl_settings_persist"]["bl_risk_profile_ui"] = str(risk_profile)
    except Exception:
        pass
    st.markdown("#### ブレンド（BL posterior と過去平均）")
    bl_alpha = st.slider(
        "α（μ_used = α*μ_BL + (1-α)*μ_hist）",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("_bl_alpha", 1.0)),
        step=0.05,
        key="bl_alpha_ui",
    )
    st.session_state["_bl_alpha"] = float(bl_alpha)
    try:
        st.session_state["bl_settings_persist"]["bl_alpha_ui"] = float(bl_alpha)
    except Exception:
        pass

    st.markdown("#### Views（主観）")
    bl_use_views = st.checkbox("Viewsを使う（絶対ビュー/相対ビュー）", value=bool(st.session_state.get("bl_settings_persist", {}).get("bl_use_views_ui", False)), key="bl_use_views_ui")
    try:
        st.session_state["bl_settings_persist"]["bl_use_views_ui"] = bool(bl_use_views)
    except Exception:
        pass


    # τは解釈が難しく、一般ユーザーが触る価値が薄いので固定値にする
    bl_tau = 0.025
    st.caption("τ（Viewsの強さ）は 0.025 固定（非表示）")
    st.session_state["_bl_tau"] = float(bl_tau)
    # --- 絶対ビュー（年率期待リターン） ---
    st.markdown("#### 絶対ビュー（各銘柄の年率期待リターン）")
    st.caption("チェックした銘柄だけに「年率期待リターン」を与えます。未チェックは prior に従います。")
    views_dict: dict[str, float] = {}
    views_conf_dict: dict[str, float] = {}

    # 候補は「今選択中のPF銘柄」（display_name→ticker変換済みを session_state に保持）
    _abs_candidates: list[str] = list(st.session_state.get("_pf_view_assets", []))
    if not _abs_candidates:
        try:
            _abs_candidates = list(dict.fromkeys(list(selected_df["ticker"].astype(str).values)))
        except Exception:
            _abs_candidates = []

    if bool(st.session_state.get("bl_use_views_ui", False)) and _abs_candidates:
        for _tkr in _abs_candidates:
            use_this = st.checkbox(f"{_tkr} に絶対ビューを付与", value=False, key=f"bl_abs_use_{_tkr}")
            if use_this:
                col_ret, col_conf = st.columns([2, 1])
                with col_ret:
                    v = st.number_input(
                        f"{_tkr} の年率期待リターン（例: 0.08=8%）",
                        value=0.05,
                        step=0.01,
                        format="%.4f",
                        key=f"bl_abs_ret_{_tkr}",
                    )
                with col_conf:
                    _conf_labels = ["低", "中", "高"]
                    _conf_map = {"低": 0.35, "中": 0.60, "高": 0.85}
                    conf_label = st.selectbox(
                        "自信度",
                        options=_conf_labels,
                        index=1,
                        key=f"bl_abs_conf_label_{_tkr}",
                        help="高いほどビューが強く反映されます。",
                    )
                    conf = float(_conf_map[str(conf_label)])
                views_dict[_tkr] = float(v)
                views_conf_dict[_tkr] = float(conf)
# --- 相対ビュー（A > B） ---
    st.markdown("#### 相対ビュー（A > B など）")
    st.caption("数値（年率差）を直接入力せず、'> / >> / >>>' の強さで指定できます。自信度が高いほどビューが強く反映されます。")
    use_rel_views = st.checkbox("相対ビューを使う", value=False, key="bl_use_rel_views")

    rel_mode = st.selectbox(
        "相対ビューの差（Q）の作り方",
        options=["固定（>→0.5%/年、>>→2%/年、>>>→5%/年）", "自動（学習データのボラに比例）"],
        index=0,
        key="bl_rel_mode",
    )
    rel_q_base = st.number_input(
        "相対ビューの基準スケール（倍率）",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state.get("_bl_rel_q_base", 1.0)),
        step=0.1,
        format="%.2f",
        key="bl_rel_q_base",
    )
    n_rel = st.slider("相対ビューの本数", min_value=0, max_value=10, value=0, step=1, key="bl_rel_n")

    _rel_candidates = _abs_candidates if _abs_candidates else ["(no assets)"]
    rel_strength_options = [">", ">>", ">>>"]
    rel_views_list: list[dict] = []

    if use_rel_views and n_rel > 0 and _rel_candidates and _rel_candidates[0] != "(no assets)":
        for i in range(int(n_rel)):
            st.markdown(f"**Relative View {i+1}**")
            a = st.selectbox("A（強気）", options=_rel_candidates, index=0, key=f"bl_rel_a_{i}")
            strength = st.selectbox("強さ（> / >> / >>>）", options=rel_strength_options, index=0, key=f"bl_rel_strength_{i}")
            b = st.selectbox("B（弱気）", options=_rel_candidates, index=min(1, len(_rel_candidates) - 1), key=f"bl_rel_b_{i}")

            _conf_labels = ["低", "中", "高"]
            _conf_map = {"低": 0.35, "中": 0.60, "高": 0.85}
            conf_label = st.selectbox(
                "自信度",
                options=_conf_labels,
                index=1,
                key=f"bl_rel_conf_label_{i}",
                help="高いほどビューが強く反映されます。",
            )
            conf = float(_conf_map[str(conf_label)])
            if a != b:
                rel_views_list.append(
                    {"a": str(a), "b": str(b), "strength": str(strength), "confidence": float(conf)}
                )

    # store for optimizer
    st.session_state["_bl_views"] = views_dict
    st.session_state["_bl_views_conf"] = views_conf_dict if bool(st.session_state.get("bl_use_views_ui", False)) else {}
    st.session_state["_bl_rel_views"] = rel_views_list if use_rel_views else []
    st.session_state["_bl_rel_mode"] = str(rel_mode)
    st.session_state["_bl_rel_q_base"] = float(rel_q_base)

# ----- ウォークフォワード（条件リバランス） -----

st.sidebar.markdown("---")
st.sidebar.subheader("ウォークフォワード（条件リバランス）")

# Read BL settings once; pass explicitly into cached optimizer to ensure correct cache keys.
bl_settings = read_bl_settings()

enable_walkforward = st.sidebar.checkbox("ウォークフォワードを有効化", value=False)
wf_method = st.sidebar.selectbox("手法", options=["MaxSharpe", "MinVol", "BL_MaxSharpe", "BL_MinVol"], index=0)
wf_opt_freq = st.sidebar.selectbox(
    "目標ウェイト更新（最適化）頻度", options=["週次(W)", "月次(M)", "年次(Y)"], index=1
)
wf_lookback_years = st.sidebar.slider(
    "学習窓（直近◯年）", min_value=1, max_value=10, value=3, step=1
)
wf_apply_tax = st.sidebar.checkbox("税金を考慮（売却益 20.315%）", value=True)

wf_threshold_rel = st.sidebar.number_input(
    "WF しきい値（相対ズレ）",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05,
    format="%.3f",
)
wf_cooldown_days = st.sidebar.number_input(
    "WF 最短間隔（営業日）0で無効",
    min_value=0,
    max_value=260,
    value=20,
    step=1,
)

wf_min_target_for_rel = st.sidebar.number_input(
    "WF 相対ズレの分母下限（目標ウェイト下限）例: 0.01 = 1%",
    min_value=0.0001,
    max_value=0.20,
    value=0.01,
    step=0.001,
    format="%.4f",
)

st.sidebar.markdown("---")
st.sidebar.caption("ウェイトは下で入力（合計が1でなくても自動正規化します）")

# =========================
# Validate selection
# =========================
if len(selected_names) == 0:
    st.warning("銘柄（ポートフォリオ）を選択してください。")
    st.stop()

selected_df = df_inst[df_inst["display_name"].isin(selected_names)].copy()
selected_df = selected_df.set_index("display_name").loc[selected_names].reset_index()

benchmark_df = None
if benchmark_name != "なし":
    benchmark_df = df_bench_all[df_bench_all["display_name"] == benchmark_name].copy()
    if benchmark_df.empty:
        st.error("ベンチマークが候補一覧に見つかりません。")
        st.stop()

# =========================
# Weight inputs
# =========================
st.subheader("ウェイト設定（ポートフォリオ）")


# ---- Apply optimized weights to current weight inputs (two-phase to avoid widget mutation errors) ----
# When a user clicks "apply weights" under an OPT table, we store the request and rerun.
# On the next run (before number_input widgets are created), we push the weights into w_<display_name> keys.
_apply_req = st.session_state.pop("_apply_weights_request", None)
if isinstance(_apply_req, dict) and _apply_req.get("weights"):
    try:
        _weights_by_ticker = dict(_apply_req.get("weights") or {})
        # Map current selection ticker -> display_name
        _dn_to_tk = selected_df.set_index("display_name")["ticker"].astype(str).to_dict()
        _tk_to_dn = {v: k for k, v in _dn_to_tk.items()}
        _tks = [str(t) for t in selected_df["ticker"].astype(str).tolist()]
        import numpy as _np
        _w = _np.array([float(_weights_by_ticker.get(t, 0.0) or 0.0) for t in _tks], dtype=float)
        _w = _np.clip(_w, 0.0, None)
        if float(_w.sum()) <= 0:
            _w[:] = 1.0 / len(_tks)
        else:
            _w = _w / float(_w.sum())
        # write to widget keys BEFORE creation
        for i, t in enumerate(_tks):
            dn = _tk_to_dn.get(t)
            if dn:
                st.session_state[f"w_{dn}"] = float(_w[i])
        # also keep in portfolio_weights
        st.session_state["portfolio_weights"] = {t: float(_w[i]) for i, t in enumerate(_tks)}
        st.success(f"ウェイトを反映しました: {_apply_req.get('label','')}")
    except Exception as _e:
        st.warning(f"ウェイト反映に失敗しました: {_e}")

# Equalize weights button
if not readonly:
    col_eq1, _ = st.columns([1, 3])
    with col_eq1:
        if st.button(
            "ウェイトを均一化",
            use_container_width=True,
            help="選択中の銘柄ウェイトをすべて同じ値（1/N）にリセットします。",
            key="btn_equalize_weights",
        ):
            eq = 1.0 / float(len(selected_names))
            for _dn in selected_names:
                st.session_state[f"w_{_dn}"] = eq
            st.rerun()
else:
    st.caption("閲覧専用モードではウェイト変更はできません。")

cols = st.columns(min(3, len(selected_names)))
w_dict: dict[str, float] = {}
_weights_by_ticker_ui = st.session_state.get("portfolio_weights", {})
if not isinstance(_weights_by_ticker_ui, dict):
    _weights_by_ticker_ui = {}
_dn_to_ticker_ui = selected_df.set_index("display_name")["ticker"].astype(str).to_dict()
for i, dn in enumerate(selected_names):
    with cols[i % len(cols)]:
        _tkr = _dn_to_ticker_ui.get(dn, "")
        _def = float(_weights_by_ticker_ui.get(_tkr, 1.0 / len(selected_names)) or 0.0)
        w_dict[dn] = st.number_input(
            dn,
            min_value=0.0,
            value=_def,
            step=0.01,
            format="%.4f",
            key=f"w_{dn}",
        )

w_arr = np.array([w_dict[dn] for dn in selected_names], dtype=float)
if np.any(w_arr < 0) or w_arr.sum() <= 0:
    st.error("ウェイトが不正です（負の値、または合計0）。")
    st.stop()
w_arr = w_arr / w_arr.sum()
weights = pd.Series(w_arr, index=selected_df["ticker"].values)

# keep current normalized weights in session_state (for save/load and exports)
try:
    st.session_state["portfolio_weights"] = {str(selected_df.iloc[i]["ticker"]): float(w_arr[i]) for i in range(len(selected_df))}
except Exception:
    pass

# =========================
# Tickers to fetch
# =========================
portfolio_tickers = selected_df["ticker"].tolist()
all_tickers = portfolio_tickers.copy()
if benchmark_df is not None:
    bench_ticker = benchmark_df.iloc[0]["ticker"]
    if bench_ticker not in all_tickers:
        all_tickers.append(bench_ticker)
rep_tickers = [x["ticker"] for x in REP_MARKET_UNIVERSE]
for rt in rep_tickers:
    if rt not in all_tickers:
        all_tickers.append(rt)
tickers = tuple(all_tickers)

meta_df = selected_df.copy()
if benchmark_df is not None:
    meta_df = pd.concat([meta_df, benchmark_df], ignore_index=True)

# Ensure representative market tickers (used for BL market model) have currency metadata for JPY conversion
rep_meta_df = pd.DataFrame(REP_MARKET_UNIVERSE)
# Add minimal columns for consistency if absent
for col, default in [("asset_type", "MARKET_PROXY"), ("bl_market_role", "MARKET_PROXY")]:
    if col not in rep_meta_df.columns:
        rep_meta_df[col] = default
missing_rep = [t for t in rep_meta_df["ticker"].tolist() if t not in set(meta_df["ticker"].tolist())]
if missing_rep:
    meta_df = pd.concat([meta_df, rep_meta_df[rep_meta_df["ticker"].isin(missing_rep)]], ignore_index=True)

# =========================
# Fetch range: include training + WF lookback buffer
# =========================
try:
    sd = pd.to_datetime(start_date)
    wf_lb = pd.DateOffset(years=max(3, int(wf_lookback_years)))
    wf_fetch_start = sd - wf_lb - pd.Timedelta(days=10)
except Exception:
    wf_fetch_start = pd.to_datetime(train_start_date)

try:
    fetch_start_dt = min(
        pd.to_datetime(train_start_date), pd.to_datetime(start_date), wf_fetch_start
    )
except Exception:
    fetch_start_dt = pd.to_datetime("2000-01-01")

_req_end_dt = pd.to_datetime(end_date)

# Session anchors to avoid re-fetching the same historical data on every period switch.
# - start anchor: only moves earlier when needed
# - end anchor: only moves later when needed
try:
    _start_anchor_raw = st.session_state.get("_fetch_start_anchor")
    _start_anchor = pd.to_datetime(_start_anchor_raw) if _start_anchor_raw else fetch_start_dt
except Exception:
    _start_anchor = fetch_start_dt
try:
    _end_anchor_raw = st.session_state.get("_fetch_end_anchor")
    _end_anchor = pd.to_datetime(_end_anchor_raw) if _end_anchor_raw else _req_end_dt
except Exception:
    _end_anchor = _req_end_dt

_start_anchor = min(_start_anchor, fetch_start_dt)
_end_anchor = max(_end_anchor, _req_end_dt)

st.session_state["_fetch_start_anchor"] = _start_anchor.strftime("%Y-%m-%d")
st.session_state["_fetch_end_anchor"] = _end_anchor.strftime("%Y-%m-%d")

fetch_start = _start_anchor.strftime("%Y-%m-%d")
fetch_end = (_end_anchor + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
with st.spinner("データ取得中..."):
    prices_local = fetch_adjclose(tickers, fetch_start, fetch_end)
    usdjpy = fetch_usdjpy(fetch_start, fetch_end)
    prices_jpy_all = to_jpy(prices_local, meta_df, usdjpy)

# Always compute using the selected end_date (even if we fetched beyond it for caching).
try:
    prices_jpy_all = prices_jpy_all.loc[prices_jpy_all.index <= _req_end_dt].copy()
except Exception:
    pass

# Cache JPY prices for BL representative market mapping (used inside optimize_weights)
st.session_state["_prices_jpy_cache"] = prices_jpy_all
st.session_state["_prices_jpy_cache_cols"] = list(prices_jpy_all.columns)

# =========================
# Evaluation alignment
# =========================
prices_jpy_pf = prices_jpy_all[portfolio_tickers].copy()
try:
    prices_jpy_pf_aligned = align_common_start(prices_jpy_pf, start_date)
except ValueError as e:
    # Keep the app usable even if the selected window has no overlapping data.
    st.error(str(e))
    st.stop()

# If the aligner decided to return an empty frame (all tickers had no data),
# surface a friendly message and stop before downstream computations explode.
if prices_jpy_pf_aligned is None or prices_jpy_pf_aligned.empty or prices_jpy_pf_aligned.shape[1] == 0:
    _d_all = getattr(prices_jpy_pf_aligned, "attrs", {}).get("dropped_no_data_tickers") if prices_jpy_pf_aligned is not None else None
    if _d_all:
        st.error(
            "指定期間に価格データが見つかりませんでした: "
            + ", ".join([str(x) for x in _d_all])
            + "\n→ 期間を広げるか、instruments.csv の ticker を確認してください。"
        )
    else:
        st.error("指定期間に価格データが見つかりませんでした。期間を広げてください。")
    st.stop()

# If some tickers had no data in the selected window, align_common_start drops them.
# Surface this to the user instead of hard-failing.
_dropped = getattr(prices_jpy_pf_aligned, "attrs", {}).get("dropped_no_data_tickers")
if _dropped:
    st.warning(
        "一部の銘柄は指定期間にデータが無かったため自動的に除外しました: "
        + ", ".join([str(x) for x in _dropped])
    )

# Evaluation weights must match the actually-available price columns.
# When a high-weight ticker is dropped (e.g., newly listed / Yahoo data missing),
# the remaining weights can accidentally sum to 0 and make downstream charts blank.
weights_eval = weights.reindex(prices_jpy_pf_aligned.columns).fillna(0.0).astype(float)
_w_sum = float(weights_eval.sum()) if prices_jpy_pf_aligned.shape[1] > 0 else 0.0
if _w_sum > 0:
    weights_eval = weights_eval / _w_sum
elif prices_jpy_pf_aligned.shape[1] > 0:
    st.warning(
        "ウェイト合計が0になったため、評価用のPF計算は残り銘柄の等ウェイトで代替します。"
        "（データ無しで除外された銘柄のウェイトを調整してください）"
    )
    weights_eval = pd.Series(
        1.0 / float(prices_jpy_pf_aligned.shape[1]), index=prices_jpy_pf_aligned.columns
    )

bench_price_aligned = None
if benchmark_df is not None:
    bench_ticker = benchmark_df.iloc[0]["ticker"]
    bench_price_aligned = (
        prices_jpy_all[bench_ticker].reindex(prices_jpy_pf_aligned.index).ffill()
    )

# =========================
# Individual charts
# =========================

st.subheader("個別銘柄チャート（JPY換算・規格化 Start=1）")

# Quick range buttons (update sidebar start/end; end is always today)
if not readonly:
    _now = pd.Timestamp.now()
    _today = _now.normalize()

    def _last_trading_day_guess(tickers: list[str], today: pd.Timestamp, now: pd.Timestamp) -> pd.Timestamp:
        """Best-effort last trading day.

        We do not ship a full holiday calendar. This function therefore uses a conservative
        weekday-based rule which fixes the most common failure mode: quick "1日" selection
        on weekends / before market open causing an empty window.

        - If all tickers are TSE (.T): use JST weekday and market open 09:00.
        - Otherwise: use weekday only (Mon-Fri).
        """
        tickers = [str(t) for t in tickers]
        all_tse = bool(tickers) and all(t.endswith(".T") for t in tickers)

        d = today
        # Weekend -> roll back to Friday
        while d.weekday() >= 5:
            d = d - pd.Timedelta(days=1)

        if all_tse:
            # Before open (09:00) -> use previous weekday
            if now.time() < pd.to_datetime("09:00").time():
                d2 = d - pd.Timedelta(days=1)
                while d2.weekday() >= 5:
                    d2 = d2 - pd.Timedelta(days=1)
                d = d2
        return d

    _last_td = _last_trading_day_guess(portfolio_tickers, _today, _now)

    def _shift_trading_days(d: pd.Timestamp, n: int) -> pd.Timestamp:
        d2 = pd.Timestamp(d)
        for _ in range(max(0, int(n))):
            d2 = d2 - pd.Timedelta(days=1)
            while d2.weekday() >= 5:
                d2 = d2 - pd.Timedelta(days=1)
        return d2

    # Swap order per request: display options first, quick buttons below.
    with st.expander("表示オプション", expanded=False):
        _log_price = st.checkbox("対数スケール（Y軸）", value=False, key="opt_individual_logy")

    st.caption("期間のクイック切替はチャート上のボタンで行えます（個別チャートのみ・高速）。")
else:
    # readonly: keep the existing expander UX
    with st.expander("表示オプション", expanded=False):
        _log_price = st.checkbox("対数スケール（Y軸）", value=False, key="opt_individual_logy")

# 1つの図に全銘柄を重ね描き（学習/評価期間の選択銘柄）
# 規格化：各系列を開始時点=1.0に揃える
meta_name_map = selected_df.set_index("ticker")["display_name"].to_dict()

try:
    # -------------------------
    # Dynamic granularity (individual chart only)
    # - >= 1 month: daily (use already-fetched JPY prices)
    # - < 1 month : intraday with appropriate interval
    #   and filter to market open hours per market.
    #
    # IMPORTANT:
    #  - For short windows (1D/5D), forcing a "common start" alignment can easily drop
    #    US tickers when timezones differ. For the individual chart we therefore normalize
    #    each series independently and merge on a union time axis for display.
    # -------------------------
    _sd_ind = pd.to_datetime(start_date)
    _ed_ind = pd.to_datetime(end_date)
    _range_days = max(1, int((_ed_ind - _sd_ind).days) + 1)
    _use_intraday = _range_days < 31
    _all_tse = bool(portfolio_tickers) and all(str(t).endswith(".T") for t in portfolio_tickers)
    _all_us = bool(portfolio_tickers) and all(not str(t).endswith(".T") for t in portfolio_tickers)
    _mixed_markets = bool(portfolio_tickers) and (not _all_tse) and (not _all_us)

    def _to_tz(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
        """Best-effort tz conversion for market-hours filtering.

        NOTE:
          Our fetcher normalizes indices to tz-naive (tz_localize(None)). In that case,
          the timestamps are already in the exchange's local time and cannot be reliably
          converted. Therefore:
            - tz-aware index: convert
            - tz-naive index: return as-is
        """
        try:
            if getattr(idx, "tz", None) is None:
                return idx
            return idx.tz_convert(tz)
        except Exception:
            return pd.DatetimeIndex(pd.to_datetime(idx))

    def _tse_hours_mask(idx: pd.DatetimeIndex) -> np.ndarray:
        idx2 = _to_tz(idx, "Asia/Tokyo")
        t = idx2.time
        m1 = (t >= pd.to_datetime("09:00").time()) & (t <= pd.to_datetime("11:30").time())
        m2 = (t >= pd.to_datetime("12:30").time()) & (t <= pd.to_datetime("15:30").time())
        return np.asarray(m1 | m2, dtype=bool)

    def _us_hours_mask(idx: pd.DatetimeIndex) -> np.ndarray:
        # US regular session: 09:30-16:00 ET
        idx2 = _to_tz(idx, "US/Eastern")
        t = idx2.time
        m = (t >= pd.to_datetime("09:30").time()) & (t <= pd.to_datetime("16:00").time())
        return np.asarray(m, dtype=bool)

    def _normalize_intraday_index(idx: pd.DatetimeIndex, tz_name: str) -> pd.DatetimeIndex:
        """Normalize intraday timestamps into market-local tz-naive times.

        yfinance may return intraday timestamps as tz-aware UTC or tz-naive UTC/local depending on ticker/provider.
        We try both interpretations (treat tz-naive as local vs as UTC) and pick the one that yields more points
        inside the market session window.
        """

        def _session_mask_hours(times: np.ndarray, market_tz: str) -> np.ndarray:
            if market_tz == "Asia/Tokyo":
                t0930 = pd.to_datetime("09:00").time()
                t1130 = pd.to_datetime("11:30").time()
                t1230 = pd.to_datetime("12:30").time()
                t1530 = pd.to_datetime("15:30").time()
                m1 = (times >= t0930) & (times <= t1130)
                m2 = (times >= t1230) & (times <= t1530)
                return np.asarray(m1 | m2, dtype=bool)
            # US/Eastern
            t0930 = pd.to_datetime("09:30").time()
            t1600 = pd.to_datetime("16:00").time()
            return np.asarray((times >= t0930) & (times <= t1600), dtype=bool)

        idx = pd.DatetimeIndex(pd.to_datetime(idx))
        if getattr(idx, "tz", None) is not None:
            try:
                return idx.tz_convert(tz_name).tz_localize(None)
            except Exception:
                try:
                    return idx.tz_localize(None)
                except Exception:
                    return idx

        # Candidate A: interpret tz-naive as market-local
        try:
            cand_local = idx.tz_localize(tz_name, ambiguous="infer", nonexistent="shift_forward").tz_convert(tz_name)
        except Exception:
            cand_local = None
        # Candidate B: interpret tz-naive as UTC
        try:
            cand_utc = idx.tz_localize("UTC").tz_convert(tz_name)
        except Exception:
            cand_utc = None

        def _score(cand) -> int:
            try:
                if cand is None or len(cand) == 0:
                    return 0
                t = cand.time
                return int(np.sum(_session_mask_hours(t, tz_name)))
            except Exception:
                return 0

        s_local = _score(cand_local)
        s_utc = _score(cand_utc)
        best = cand_utc if s_utc > s_local else cand_local
        if best is None:
            return idx
        try:
            return best.tz_localize(None)
        except Exception:
            return idx

    def _densify_intraday_series_5m(s: pd.Series, market: str) -> pd.Series:
        """Densify intraday series to a 5-min grid within market sessions (session-local).

        Purpose: Some tickers (esp. low-liquidity) have missing 5-min bars from Yahoo,
        which renders as broken lines. Within trading sessions, we forward-fill on a
        5-min grid to keep the line continuous while still leaving non-session gaps empty.
        """
        try:
            if s is None:
                return s
            s = s.dropna()
            if s.shape[0] < 2:
                return s
            s = s.sort_index()
            try:
                s = s[~s.index.duplicated(keep="last")]
            except Exception:
                pass

            # Snap to 5-min grid (handles odd timestamps)
            try:
                _idx_floor = pd.DatetimeIndex(pd.to_datetime(s.index)).floor("5min")
                s = s.groupby(_idx_floor).last()
            except Exception:
                pass

            if market == "TSE":
                _sessions = [("09:00", "11:30"), ("12:30", "15:30")]
            else:
                _sessions = [("09:30", "16:00")]

            days = pd.DatetimeIndex(pd.to_datetime(s.index).normalize()).unique()
            if len(days) == 0:
                return s
            days = pd.DatetimeIndex(days).sort_values()

            parts: list[pd.Series] = []
            for d in days:
                dd = pd.Timestamp(d).date()
                for a, b in _sessions:
                    try:
                        t0 = pd.Timestamp.combine(dd, pd.to_datetime(a).time())
                        t1 = pd.Timestamp.combine(dd, pd.to_datetime(b).time())
                        idx5 = pd.date_range(t0, t1, freq="5min")
                        part = s.reindex(idx5).ffill()
                        parts.append(part)
                    except Exception:
                        continue

            if not parts:
                return s
            out = pd.concat(parts).sort_index()
            try:
                out = out[~out.index.duplicated(keep="last")]
            except Exception:
                pass
            return out
        except Exception:
            return s

    if _use_intraday:
        # Choose a sensible interval by range (Yahoo limits apply; keep conservative)
        #  - <= 7d can use finer intervals; beyond that, go coarser.
        if _range_days <= 2:
            _itv = "5m"
        elif _range_days <= 7:
            _itv = "5m"
        else:
            _itv = "60m"

        # Fetch only the portfolio tickers for the chart (avoid perturbing PF analytics)
        # For very short windows (1D/5D), keep a fixed fetch-start buffer so
        # switching between 1日 and 5日 can hit the cache (same args).
        _sd_fetch = _sd_ind
        if _range_days <= 7:
            _sd_fetch = _ed_ind - pd.Timedelta(days=7)

        _fetch_end = (_ed_ind + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        _px_local_i = fetch_adjclose(
            tuple(portfolio_tickers),
            _sd_fetch.strftime("%Y-%m-%d"),
            _fetch_end,
            interval=_itv,
        )
        _px_jpy_i = to_jpy(_px_local_i, meta_df, usdjpy)

        # Trim to the requested window first
        _px_jpy_i = _px_jpy_i.loc[(_px_jpy_i.index >= _sd_ind) & (_px_jpy_i.index <= _ed_ind + pd.Timedelta(days=1))].copy()

        # Apply market-hours filtering per ticker. If filtering removes too much, keep unfiltered.
        _series_norm: dict[str, pd.Series] = {}
        _union_idx = None

        for _t in list(_px_jpy_i.columns):
            _s0 = _px_jpy_i[_t].dropna()
            if _s0.empty:
                continue

            if str(_t).endswith(".T"):
                _s0.index = _normalize_intraday_index(_s0.index, "Asia/Tokyo")
                _mask = _tse_hours_mask(_s0.index)
            else:
                _s0.index = _normalize_intraday_index(_s0.index, "US/Eastern")
                _mask = _us_hours_mask(_s0.index)

            _s1 = _s0.loc[_mask] if _mask is not None else _s0
            # If mask kills most points (timezone heuristics can be wrong), fall back to unfiltered
            if _s1.shape[0] < 2 and _s0.shape[0] >= 2:
                _s1 = _s0

            if _s1.shape[0] < 2:
                continue

            _s1 = _s1.sort_index()
            if _itv == "5m":
                _s1 = _densify_intraday_series_5m(_s1, "TSE" if str(_t).endswith(".T") else "US")
            _s1 = _s1 / float(_s1.iloc[0])
            _series_norm[str(_t)] = _s1
            _union_idx = _s1.index if _union_idx is None else _union_idx.union(_s1.index)

        if _union_idx is None or len(_union_idx) < 2:
            _use_intraday = False
        else:
            _union_idx = _union_idx.sort_values()
            _px_df = pd.DataFrame(index=_union_idx)
            for _t, _s in _series_norm.items():
                # Mixed markets: do NOT fill missing points across the other market's timestamps.
                # This keeps "out-of-session" periods visually absent instead of flat lines.
                if _mixed_markets:
                    _px_df[_t] = _s.reindex(_union_idx)
                else:
                    _px_df[_t] = _s.reindex(_union_idx).ffill()

    if not _use_intraday:
        # Daily: normalize each series independently (avoid dropping US tickers on 1D/5D windows)
        _px_df0 = prices_jpy_pf.loc[(prices_jpy_pf.index >= _sd_ind) & (prices_jpy_pf.index <= _ed_ind)].copy()
        _norm_series: dict[str, pd.Series] = {}
        _union_idx = None
        for _t in list(_px_df0.columns):
            _s = _px_df0[_t].dropna()
            if _s.shape[0] < 1:
                continue
            _s = _s.sort_index()
            _s = _s / float(_s.iloc[0])
            _norm_series[str(_t)] = _s
            _union_idx = _s.index if _union_idx is None else _union_idx.union(_s.index)
        if _union_idx is None or len(_union_idx) == 0:
            _px_df = pd.DataFrame()
        else:
            _union_idx = _union_idx.sort_values()
            # If the selected window yields only a single daily point (common for 1D on weekends/holidays),
            # duplicate the point at the end of the window so the plot/table can still render.
            if len(_union_idx) == 1:
                _union_idx = _union_idx.union([_union_idx[0] + pd.Timedelta(hours=23, minutes=59)]).sort_values()

            _px_df = pd.DataFrame(index=_union_idx)
            for _t, _s in _norm_series.items():
                _px_df[_t] = _s.reindex(_union_idx).ffill()

    if _px_df.shape[0] >= 2 and _px_df.shape[1] >= 1:
        _norm = _px_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        # long-form
        _df_long = _norm.reset_index()
        # index列名が何であっても先頭列をdateに統一
        _date_col = _df_long.columns[0]
        if _date_col != "date":
            _df_long = _df_long.rename(columns={_date_col: "date"})
        _df_long = _df_long.melt(id_vars="date", var_name="ticker", value_name="value")
        _df_long["name"] = _df_long["ticker"].map(lambda t: f"{meta_name_map.get(t, t)} ({t})")

        # Build a multi-series chart (one trace per ticker) with fixed colors.
        # We hide the on-chart legend (requested) and show a legend+%change list directly under the chart.
        fig_prices = go.Figure()
        _tickers = list(_norm.columns)
        _palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        _color_map: dict[str, str] = {}
        for i, t in enumerate(_tickers):
            c = _palette[i % len(_palette)]
            _color_map[str(t)] = c
            nm = f"{meta_name_map.get(t, t)} ({t})"
            fig_prices.add_trace(
                go.Scatter(
                    x=_norm.index,
                    y=_norm[t].values,
                    mode="lines",
                    name=nm,
                    line=dict(color=c, width=2),
                    showlegend=False,
                )
            )

        fig_prices.update_layout(
            title="Individual Prices (JPY) Normalized (Start=1.0)",
            hovermode="x",
            autosize=True,
            margin=dict(l=10, r=10, t=50, b=50),
            title_font=dict(color="rgba(49,51,63,1)")
        )
        fig_prices.update_traces(hoverinfo="skip", hovertemplate=None)
        _x_rangebreaks = None
        if _use_intraday and (_all_tse or _all_us):
            if _all_tse:
                _x_rangebreaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[15.5, 9.0], pattern="hour"),
                    dict(bounds=[11.5, 12.5], pattern="hour"),
                ]
            else:
                _x_rangebreaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16.0, 9.5], pattern="hour"),
                ]

        _xaxis_kwargs = dict(
            automargin=True,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            spikethickness=1,
        )
        if _x_rangebreaks is not None:
            _xaxis_kwargs["rangebreaks"] = _x_rangebreaks
            _xaxis_kwargs["range"] = [pd.Timestamp(_norm.index.min()), pd.Timestamp(_norm.index.max())]
        fig_prices.update_xaxes(**_xaxis_kwargs)
        fig_prices.update_yaxes(title_text="Normalized Price (Start=1.0)", tickformat=".2f", automargin=True)

        if _log_price:
            fig_prices.update_yaxes(type="log")

        # Front-only Level-2 UX: chart + table are updated in the component without
        # triggering Streamlit reruns on hover.
        # Also include:
        #  - daily (wide window) raw series for fast client-side range switching
        #  - intraday (recent) raw series for 1D/5D buttons without reruns
        _daily_payload = {}
        _intraday_payload = {}
        try:
            _ed_for_payload = pd.to_datetime(end_date)
            # For frontend quick-range (including "最大"), fetch a wide daily history
            # independent of the sidebar eval window.
            _daily_start = "2000-01-01"
            _daily_end = (_ed_for_payload + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            _daily_local = fetch_adjclose(tuple(portfolio_tickers), _daily_start, _daily_end, interval="1d")
            _daily_fx = fetch_usdjpy(_daily_start, _daily_end)
            _daily_raw = to_jpy(_daily_local, meta_df, _daily_fx).sort_index()
            _daily_raw = _daily_raw.reindex(columns=portfolio_tickers).ffill()
            if _daily_raw.shape[0] >= 2 and _daily_raw.shape[1] >= 1:
                _daily_payload = {
                    "x": [ts.isoformat() for ts in _daily_raw.index],
                    "series": {str(t): _daily_raw[t].astype(float).values for t in _daily_raw.columns},
                }
        except Exception:
            _daily_payload = {}

        try:
            _now_ts = pd.Timestamp.now()
            _ed_user = pd.to_datetime(end_date)
            # Intraday data from Yahoo is only available for a limited recent window.
            # Also, passing an end-date beyond "now" can yield empty intraday results.
            _end_intr = min(_now_ts, _ed_user + pd.Timedelta(days=1))
            if (_now_ts - _end_intr) > pd.Timedelta(days=60):
                raise RuntimeError("intraday out of supported range")

            _sd_recent = _end_intr - pd.Timedelta(days=7)
            _px_local_r = fetch_adjclose(
                tuple(portfolio_tickers),
                _sd_recent,
                _end_intr,
                interval="5m",
            )
            _px_jpy_r = to_jpy(_px_local_r, meta_df, usdjpy)
            _px_jpy_r = _px_jpy_r.sort_index().copy()
            _px_jpy_r = _px_jpy_r.reindex(columns=portfolio_tickers)

            _series_raw_r: dict[str, pd.Series] = {}
            _union_idx_r = None
            _has_tse_r = any(str(t).endswith(".T") for t in portfolio_tickers)
            _has_us_r = any(not str(t).endswith(".T") for t in portfolio_tickers)
            _mixed_markets_r = _has_tse_r and _has_us_r
            for _t in list(_px_jpy_r.columns):
                _s0 = _px_jpy_r[_t].dropna()
                if _s0.empty:
                    continue

                if str(_t).endswith(".T"):
                    _s0.index = _normalize_intraday_index(_s0.index, "Asia/Tokyo")
                    _mask = _tse_hours_mask(_s0.index)
                else:
                    _s0.index = _normalize_intraday_index(_s0.index, "US/Eastern")
                    _mask = _us_hours_mask(_s0.index)

                _s1 = _s0.loc[_mask] if _mask is not None else _s0
                if _s1.shape[0] < 2 and _s0.shape[0] >= 2:
                    _s1 = _s0
                if _s1.shape[0] < 2:
                    continue

                _s1 = _s1.sort_index()
                _s1 = _densify_intraday_series_5m(_s1, "TSE" if str(_t).endswith(".T") else "US")
                _series_raw_r[str(_t)] = _s1
                _union_idx_r = _s1.index if _union_idx_r is None else _union_idx_r.union(_s1.index)

            if _union_idx_r is not None and len(_union_idx_r) >= 2:
                _union_idx_r = _union_idx_r.sort_values()
                _raw_df_r = pd.DataFrame(index=_union_idx_r)
                for _t, _s in _series_raw_r.items():
                    # Mixed markets (JP+US): keep each market's local clock time (tz-naive) so
                    # sessions visually overlap; also avoid forward-filling across the other
                    # market's timestamps so off-session periods remain empty.
                    if _mixed_markets_r:
                        _raw_df_r[_t] = _s.reindex(_union_idx_r)
                    else:
                        _raw_df_r[_t] = _s.reindex(_union_idx_r).ffill()
                _intraday_payload = {
                    "x": [ts.isoformat() for ts in _raw_df_r.index],
                    "series": {str(t): _raw_df_r[str(t)].astype(float).values for t in _raw_df_r.columns},
                }
        except Exception:
            _intraday_payload = {}

        _table_payload = {
            "x": [ts.isoformat() for ts in _norm.index],
            "order": [str(t) for t in _norm.columns],
            "names": {str(t): str(meta_name_map.get(t, t)) for t in _norm.columns},
            "colors": {str(k): str(v) for k, v in _color_map.items()},
            "series": {str(t): _norm[t].astype(float).values for t in _norm.columns},
            "daily": _daily_payload,
            "intraday": _intraday_payload,
        }
        # Respect sidebar graph-size controls
        render_plotly_with_table_front(
            fig_prices,
            _table_payload,
            key="prices_front",
            height=plotly_height_px,
            width=int(display_width_px),
        )
    else:
        st.info("プロットに必要なデータ点が不足しています。")
except Exception as e:
    st.warning(f"個別銘柄チャートの作成に失敗しました: {e}")

# =========================
# Base PF
# =========================
st.subheader("PF（指数化 Start=100）")

shares, pf_value, pf_index = build_buy_and_hold_pf(
    prices_jpy_pf_aligned, weights_eval, initial_value
)
metrics_bh, _, _ = calc_metrics_from_pf(
    pf_value, trading_days_per_year, rf_annual=rf_annual
)

pf_index_to_plot: dict[str, pd.Series] = {"Buy&Hold": pf_index}
metrics_table: dict[str, dict] = {"Buy&Hold": metrics_bh}

# =========================
# Fixed-weight rebalance (calendar or threshold)
# =========================
pv_rt = None
pi_rt = None
tax_paid = None
reb_flag = None
max_rel_dev = None

if rebalance_mode != "なし（買いっぱなし）":
    if rebalance_rule == "カレンダー（D/W/M/Y）":
        freq = {"日次(D)": "D", "週次(W)": "W", "月次(M)": "M", "年次(Y)": "Y"}[
            rebalance_mode
        ]
        pv_rt, pi_rt, tax_paid = build_rebalanced_pf_with_tax(
            prices_jpy_pf_aligned,
            target_weights=weights_eval,
            initial_value=initial_value,
            rebalance_freq=freq,
            tax_rate=tax_rate,
                min_target_for_rel=float(min_target_for_rel),
        )
        label = f"Rebal {freq} + Tax"
    else:
        pv_rt, pi_rt, tax_paid, reb_flag, max_rel_dev = (
            build_threshold_rebalanced_pf_with_tax(
                prices_jpy_pf_aligned,
                target_weights=weights_eval,
                initial_value=initial_value,
                threshold_rel=float(threshold_rel),
                cooldown_days=int(cooldown_days),
                tax_rate=tax_rate,
            )
        )
        label = f"Rebal REL({threshold_rel:.3f}) cd={int(cooldown_days)} + Tax"

    met_rt, _, _ = calc_metrics_from_pf(
        pv_rt, trading_days_per_year, rf_annual=rf_annual
    )
    pf_index_to_plot[label] = pi_rt
    metrics_table[label] = met_rt

# Optional: show threshold diagnostics (fixed-weight)
if max_rel_dev is not None:
    st.subheader("固定ウェイト：相対ズレ（診断）")
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    plt.plot(max_rel_dev.index, max_rel_dev.values)
    plt.axhline(float(threshold_rel), linestyle="--")
    plt.title("Max relative weight deviation (fixed-weight)")
    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        f"固定ウェイト：リバランス回数 = {int(reb_flag.sum()) if reb_flag is not None else 0}"
    )

# =========================
# Static optimization (train before start_date) -> Buy&Hold on eval
# =========================
_prev_opt_weights_raw = st.session_state.get("_opt_weights_map", {})
if not isinstance(_prev_opt_weights_raw, dict):
    _prev_opt_weights_raw = {}

# We will *update* this store only when an optimization succeeds.
# If optimization fails (e.g., network/BL inputs/solver issues), we keep the last known-good result
# so the OPT series does not disappear from the UI.
opt_weights_store: dict[str, dict] = {
    str(k): (v if isinstance(v, dict) else {}) for k, v in _prev_opt_weights_raw.items()
}

prices_train_raw = prices_jpy_all[portfolio_tickers].copy()
prices_train_raw = prices_train_raw.loc[
    (prices_train_raw.index >= pd.to_datetime(train_start_date))
    & (prices_train_raw.index <= pd.to_datetime(train_end_date))
].copy()

prices_train = None
try:
    prices_train = align_common_start(prices_train_raw, train_start_date)
    _dropped_train = getattr(prices_train, "attrs", {}).get("dropped_no_data_tickers")
    if _dropped_train:
        st.warning(
            "学習期間にデータが無い銘柄を除外しました（最適化対象外）: "
            + ", ".join([str(x) for x in _dropped_train])
        )
except Exception as e:
    if len(opt_choices) > 0:
        st.warning(f"学習データの整合に失敗しました（最適化はスキップ）: {e}")

try:
    ts = pd.to_datetime(train_start_date)
    te = pd.to_datetime(train_end_date)
    eval_s = pd.to_datetime(start_date)
    if te >= eval_s:
        st.warning(
            "学習終了日が開始日以降です。学習終了日は開始日の前日以前にしてください。"
        )
    if ts >= te:
        st.warning("学習開始日が学習終了日以降です。学習期間を見直してください。")
except Exception:
    pass

if prices_train is not None and len(opt_choices) > 0:
    for m in opt_choices:
        try:
            w_opt = optimize_weights_cached(
                prices_train,
                method=m,
                rf_annual=rf_annual,
                trading_days_per_year=trading_days_per_year,
                bl_settings=bl_settings,
            )
            # persist as plain dict (session_state-friendly)
            opt_weights_store[str(m)] = w_opt.astype(float).to_dict()
        except Exception as e:
            st.warning(f"{m} の最適化に失敗しました（スキップ）: {e}")

# Persist updated store
st.session_state["_opt_weights_map"] = opt_weights_store

# Build series map for downstream calculations / display
opt_weights_map: dict[str, pd.Series] = {}
for _name, _w_dict in opt_weights_store.items():
    if isinstance(_w_dict, dict) and len(_w_dict) > 0:
        opt_weights_map[_name] = pd.Series(_w_dict, dtype=float)

opt_pf_values: dict[str, pd.Series] = {}
opt_pf_indices: dict[str, pd.Series] = {}

# Show only currently selected optimization methods, but keep last known-good
# results for those methods even when the latest run fails.
opt_weights_map_visible: dict[str, pd.Series] = {
    str(m): opt_weights_map[str(m)] for m in opt_choices if str(m) in opt_weights_map
}

if len(opt_weights_map_visible) > 0:
    st.subheader("最適化ポートフォリオ（学習期間→評価期間に適用）")
    for name, w_opt in opt_weights_map_visible.items():
        _, pv_opt, pi_opt = build_buy_and_hold_pf(
            prices_jpy_pf_aligned,
            weights=w_opt.reindex(prices_jpy_pf_aligned.columns),
            initial_value=initial_value,
        )
        label = f"OPT: {name}"
        pf_index_to_plot[label] = pi_opt
        met_opt, _, _ = calc_metrics_from_pf(
            pv_opt, trading_days_per_year, rf_annual=rf_annual
        )
        metrics_table[label] = met_opt
        opt_pf_values[label] = pv_opt
        opt_pf_indices[label] = pi_opt

        with st.expander(f"{label} ウェイト"):
            _w_tbl = w_opt.sort_values(ascending=False).to_frame("Weight")
            _w_tbl.insert(
                0, "銘柄名", [str(meta_name_map.get(str(t), str(t))) for t in _w_tbl.index]
            )
            st.dataframe(_w_tbl, use_container_width=True)
            if not readonly:
                if st.button(
                    "このウェイトを現在の設定に反映",
                    use_container_width=True,
                    key=f"btn_apply_weights_{name}",
                    help="この最適化ウェイトをウェイト設定欄に反映します（評価期間のPF配分が変わります）。",
                ):
                    st.session_state["_apply_weights_request"] = {
                        "label": label,
                        "weights": w_opt.astype(float).to_dict(),
                    }
                    st.rerun()

# =========================
# Walk-forward (conditional rebalance)
# =========================
wf_pv = None
wf_pi = None
wf_tax_paid = None
wf_weights_hist = None
wf_reb_flag = None
wf_max_rel_dev = None
wf_label = None

if enable_walkforward:
    freq_map = {"週次(W)": "W", "月次(M)": "M", "年次(Y)": "Y"}
    wf_freq_code = freq_map[wf_opt_freq]

    prices_train_all = prices_jpy_all[portfolio_tickers].copy()

    try:
        wf_pv, wf_pi, wf_tax_paid, wf_weights_hist, wf_reb_flag, wf_max_rel_dev = (
            build_walk_forward_pf_with_tax(
                prices_eval=prices_jpy_pf_aligned,
                prices_all_for_train=prices_train_all,
                initial_value=initial_value,
                opt_freq=wf_freq_code,
                method=wf_method,
                rf_annual=rf_annual,
                trading_days_per_year=trading_days_per_year,
                lookback_years=int(wf_lookback_years),
                tax_rate=tax_rate,
                apply_tax=bool(wf_apply_tax),
                bl_settings=bl_settings,
                threshold_rel=float(wf_threshold_rel),
                cooldown_days=int(wf_cooldown_days),
                min_target_for_rel=float(wf_min_target_for_rel),
            )
        )

        wf_label = (
            f"WF: {wf_method} ({wf_freq_code}, {wf_lookback_years}y"
            f", REL={wf_threshold_rel:.3f}, cd={int(wf_cooldown_days)}"
            + (", Tax" if wf_apply_tax else "")
            + ")"
        )
        pf_index_to_plot[wf_label] = wf_pi
        met_wf, _, _ = calc_metrics_from_pf(
            wf_pv, trading_days_per_year, rf_annual=rf_annual
        )
        metrics_table[wf_label] = met_wf

    except Exception as e:
        st.warning(f"ウォークフォワードの計算に失敗しました（スキップ）: {e}")

if wf_max_rel_dev is not None:
    st.subheader("WF：相対ズレ（診断）")
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    plt.plot(wf_max_rel_dev.index, wf_max_rel_dev.values)
    plt.axhline(float(wf_threshold_rel), linestyle="--")
    plt.title("WF Max relative weight deviation")
    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        f"WF：リバランス回数 = {int(wf_reb_flag.sum()) if wf_reb_flag is not None else 0}"
    )

    if wf_weights_hist is not None and len(wf_weights_hist) > 0:
        with st.expander("WF：目標ウェイト更新履歴（抜粋）"):
            wh_df = pd.DataFrame(
                [{"date": d, **w.to_dict()} for d, w in wf_weights_hist]
            ).set_index("date")
            st.dataframe(wh_df.tail(20))

# =========================
# Benchmark alignment + metrics
# =========================
bench_index = None
bench_metrics = None
pf_value_common = None
pf_index_common = None
bench_price_common = None

if benchmark_df is not None and bench_price_aligned is not None:
    common_idx = pf_value.index.intersection(bench_price_aligned.dropna().index)
    if len(common_idx) >= 2:
        pf_value_common = pf_value.loc[common_idx]
        pf_index_common = (100.0 * pf_value_common / pf_value_common.iloc[0]).rename(
            "PF_INDEX_COMMON"
        )

        bench_price_common = bench_price_aligned.loc[common_idx].rename("BENCH_VALUE")
        bench_index = (100.0 * bench_price_common / bench_price_common.iloc[0]).rename(
            "BENCH_INDEX"
        )

        bench_metrics, _, _ = calc_metrics_from_pf(
            bench_price_common, trading_days_per_year, rf_annual=rf_annual
        )

        pf_index_to_plot[f"Benchmark: {benchmark_name}"] = bench_index
        metrics_table[f"Benchmark: {benchmark_name}"] = bench_metrics

        # 超過（PF - Bench）の簡易差分（指標）
        excess = {}
        for k in metrics_bh.keys():
            if bench_metrics and k in bench_metrics:
                excess[k] = metrics_bh[k] - bench_metrics[k]
        metrics_table["Excess (PF - Bench)"] = excess


# =========================
# PF index chart (with benchmark + OPT + WF) - Plotly
# =========================
st.subheader("PF（指数化・規格化 Start=1）")

try:
    # 共通インデックスで揃えて規格化（Start=1.0）
    _series_list = []
    for label, s in pf_index_to_plot.items():
        if s is None or len(s) < 2:
            continue
        _ss = s.dropna()
        if len(_ss) < 2:
            continue
        _ss = (_ss / _ss.iloc[0]).rename(label)  # Start=1
        _series_list.append(_ss)

    if len(_series_list) == 0:
        st.info("PF指数の描画に必要なデータが不足しています。")
    else:
        _df_pf = pd.concat(_series_list, axis=1).dropna(how="all")
        _df_long = _df_pf.reset_index()
        _date_col = _df_long.columns[0]
        if _date_col != "date":
            _df_long = _df_long.rename(columns={_date_col: "date"})
        _df_long = _df_long.melt(id_vars="date", var_name="series", value_name="value")

        title_bench = f" vs {benchmark_name}" if benchmark_df is not None else ""
        fig_pf = px.line(
            _df_long,
            x="date",
            y="value",
            color="series",
            title=f"PF Index Normalized (Start=1.0){title_bench}",
        )
        fig_pf.update_yaxes(title_text="Index (Start=1.0)")
        fig_pf = _apply_plotly_mobile_layout(fig_pf, legend_title="Series")

        # Level-1 chart UX: unified hover + vertical spike line at cursor
        fig_pf.update_layout(hovermode="x unified")
        fig_pf.update_xaxes(
            automargin=True,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            spikethickness=1,
        )

        # Respect sidebar graph-size controls
        fig_pf.update_layout(height=plotly_height_px, width=int(display_width_px))
        try:
            st.plotly_chart(fig_pf, use_container_width=False)
        except TypeError:
            st.plotly_chart(fig_pf, width="stretch")
        # Show latest % change table under the chart
        try:
            _render_change_table_under_chart(
                label="PF/ベンチマークの変化率",
                norm_df=_df_pf,
            )
        except Exception:
            pass
        
except Exception as e:
    st.warning(f"PF指数チャートの作成に失敗しました: {e}")

# =========================
# Metrics table
# =========================
# =========================
# In-sample (training period) metrics
# =========================
train_metrics_table: dict[str, dict] = {}
try:
    _ts = pd.to_datetime(train_start_date)
    _te = pd.to_datetime(train_end_date)
    train_period_str = f"{_ts.strftime('%Y-%m-%d')} ～ {_te.strftime('%Y-%m-%d')}"
except Exception:
    train_period_str = f"{train_start_date} ～ {train_end_date}"

try:
    if prices_train is not None and not prices_train.empty:
        # Align training prices to the same tickers used in the evaluation PF
        try:
            prices_train_pf = prices_train[prices_jpy_pf_aligned.columns].dropna(how="any").copy()
        except Exception:
            prices_train_pf = prices_train.dropna(how="any").copy()

        if prices_train_pf.shape[0] >= 2:
            # Buy&Hold (using the same user-specified weights)
            _, pv_bh_tr, _ = build_buy_and_hold_pf(
                prices_train_pf,
                weights=weights.reindex(prices_train_pf.columns),
                initial_value=initial_value,
            )
            met_bh_tr, _, _ = calc_metrics_from_pf(
                pv_bh_tr, trading_days_per_year, rf_annual=rf_annual
            )
            train_metrics_table["Buy&Hold"] = met_bh_tr

            # Optimized portfolios (same optimized weights, evaluated on training period)
            if len(opt_weights_map) > 0:
                for _name, _w in opt_weights_map.items():
                    try:
                        _, pv_opt_tr, _ = build_buy_and_hold_pf(
                            prices_train_pf,
                            weights=_w.reindex(prices_train_pf.columns),
                            initial_value=initial_value,
                        )
                        met_opt_tr, _, _ = calc_metrics_from_pf(
                            pv_opt_tr, trading_days_per_year, rf_annual=rf_annual
                        )
                        train_metrics_table[f"OPT: {_name}"] = met_opt_tr
                    except Exception:
                        continue
except Exception:
    # Training metrics are optional; do not fail the app if data/inputs are invalid.
    train_metrics_table = {}

st.subheader("指標")
dfm = pd.DataFrame(metrics_table)
st.dataframe(dfm)

# ---- Training-period metrics (in-sample) ----
if len(train_metrics_table) > 0:
    st.subheader("指標（学習期間）")
    st.caption(f"学習期間: {train_period_str}")
    dfm_train = pd.DataFrame(train_metrics_table)
    st.dataframe(dfm_train)

    # ---- Training-period price moves (normalized to 1 at start) ----
    with st.expander("学習期間の値動き（開始=1に規格化）", expanded=False):
        if px is None or go is None:
            st.warning("Plotly がインストールされていないためグラフ表示ができません。`pip install plotly` を実行してください。")
        else:
            try:
                _ts = pd.to_datetime(train_start_date)
                _te = pd.to_datetime(train_end_date)
                _te_inc = (_te + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                _ts_s = _ts.strftime("%Y-%m-%d")

                # Use current selection
                _sel_df = df_inst[df_inst["display_name"].isin(selected_names)].copy()
                _tickers = tuple(_sel_df["ticker"].astype(str).tolist())

                if len(_tickers) == 0:
                    st.info("銘柄が選択されていません。左の銘柄選択で追加してください。")
                else:
                    _train_px = fetch_adjclose(_tickers, start=_ts_s, end=_te_inc).dropna(how="all")
                    if _train_px.shape[0] < 2:
                        st.warning("学習期間の価格データが不足しています。期間や銘柄を見直してください。")
                    else:
                        # Normalize to 1 at start
                        _train_px = _train_px.ffill().dropna(how="any")
                        _norm = _train_px / _train_px.iloc[0]
                        # Ensure we always have a 'date' column after reset_index(), regardless of index name.
                        _norm = _norm.reset_index()
                        _date_col = _norm.columns[0]
                        if _date_col != "date":
                            _norm = _norm.rename(columns={_date_col: "date"})

                        # Long-form for plotly
                        _long = _norm.melt(id_vars="date", var_name="ticker", value_name="normalized")
                        # Map ticker -> display_name for readability
                        _t2n = dict(zip(_sel_df["ticker"].astype(str), _sel_df["display_name"].astype(str)))
                        _long["name"] = _long["ticker"].map(_t2n).fillna(_long["ticker"])

                        _use_log = st.checkbox("対数スケールで表示（相対変化を見やすく）", value=False, key="train_norm_log")
                        fig = px.line(_long, x="date", y="normalized", color="name")
                        # Mobile-friendly legend (place below chart)
                        fig = _apply_plotly_mobile_layout(fig, legend_title="")
                        # Level-1 chart UX: unified hover + vertical spike line at cursor
                        fig.update_layout(hovermode="x unified")
                        fig.update_xaxes(
                            automargin=True,
                            showspikes=True,
                            spikemode="across",
                            spikesnap="cursor",
                            spikedash="dot",
                            spikethickness=1,
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Normalized (Start=1.0)",
                            height=plotly_height_px,
                            width=int(display_width_px),
                        )
                        if _use_log:
                            fig.update_yaxes(type="log")

                        try:
                            st.plotly_chart(fig, use_container_width=False)
                        except TypeError:
                            st.plotly_chart(fig, width="stretch")


                        with st.expander("学習期間データ（規格化後）を表示", expanded=False):
                            _wide = _norm.set_index("date")
                            st.dataframe(_wide, use_container_width=True)
            except Exception as e:
                st.warning(f"学習期間の値動きグラフの作成に失敗しました: {e}")



# =========================
# BL prior π sensitivity to δ (risk aversion)
# =========================
try:
    _pi_pack = st.session_state.get("_bl_pi_sensitivity", None)
    if isinstance(_pi_pack, dict) and isinstance(_pi_pack.get("pi", None), pd.DataFrame):
        with st.expander("BL：市場均衡リターン π のδ感度（リスク許容度による変化）", expanded=False):
            _pi_df = _pi_pack["pi"].copy()
            _mm = str(_pi_pack.get("market_model", ""))
            _dcur = float(_pi_pack.get("delta_current", np.nan))
            st.caption(
                f"市場モデル: {_mm} / 現在選択のδ: {_dcur:.3g}。"
                " 表は年率の市場均衡リターン（π）で、δを変更したときにどれだけスケールするか（または写像で変形するか）を示します。"
            )

            _pi_pct = (_pi_df * 100.0).round(3)
            st.dataframe(_pi_pct, use_container_width=True)

            # Optional: single-asset curve view
            _assets = list(_pi_df.index)
            if len(_assets) > 0:
                _pick = st.selectbox("銘柄を選んで π（年率）を曲線で確認", options=_assets, index=0, key="pi_sens_asset_pick")
                _s = (_pi_df.loc[_pick] * 100.0).astype(float)
                _x = list(range(len(_s.index)))
                fig = plt.figure(figsize=(fig_width, max(3, int(fig_height * 0.8))), dpi=fig_dpi)
                plt.plot(_x, _s.values, marker="o")
                plt.title(f"π sensitivity: {_pick}")
                plt.xlabel("δプロファイル（積極→標準→保守）")
                plt.ylabel("π (annual, %)")
                plt.xticks(_x, list(_s.index), rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
except Exception:
    pass



# =========================
# BL representative market betas (stocks x ETFs)
# =========================
try:
    _b_pack = st.session_state.get("_bl_rep_betas", None)
    if isinstance(_b_pack, dict) and isinstance(_b_pack.get("betas", None), pd.DataFrame):
        with st.expander("BL：代表ETF市場のβ可視化（銘柄×ETF）", expanded=False):
            _b_df = _b_pack["betas"].copy()
            _mm = str(_b_pack.get("market_model", ""))
            st.caption(
                "学習期間のリターンで、各銘柄のリターンを代表ETF群のリターンで多変量回帰した係数（β）です。"
                " 『どのETFに属するか』ではなく、『どのETF要因にどれだけ反応するか』を示します。"
                f" 市場モデル: {_mm}"
            )

            # Controls
            _abs = st.checkbox("絶対値で色付け（符号は維持）", value=True, key="beta_heat_abs")
            _sort_by = st.selectbox(
                "並び替え基準（銘柄）",
                options=["表示順", "最大|β|が大きい順", "合計|β|が大きい順"],
                index=0,
                key="beta_sort_assets",
            )

            _b_for_sort = _b_df.copy()
            if _sort_by == "最大|β|が大きい順":
                _order = _b_for_sort.abs().max(axis=1).sort_values(ascending=False).index
                _b_for_sort = _b_for_sort.loc[_order]
            elif _sort_by == "合計|β|が大きい順":
                _order = _b_for_sort.abs().sum(axis=1).sort_values(ascending=False).index
                _b_for_sort = _b_for_sort.loc[_order]

            # Plotly heatmap (preferred)
            if go is not None:
                _z = _b_for_sort.values
                _z_show = np.abs(_z) if _abs else _z
                fig = go.Figure(
                    data=go.Heatmap(
                        z=_z_show,
                        x=list(_b_for_sort.columns),
                        y=list(_b_for_sort.index),
                        colorbar=dict(title="|β|" if _abs else "β"),
                        hovertemplate="Asset=%{y}<br>ETF=%{x}<br>β=%{customdata:.4f}<extra></extra>",
                        customdata=_z,
                    )
                )
                fig.update_layout(
                    height=min(900, 260 + 22 * len(_b_for_sort.index)),
                    width=int(display_width_px),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Representative ETFs",
                    yaxis_title="Assets",
                )
                try:
                    st.plotly_chart(fig, use_container_width=False)
                except TypeError:
                    st.plotly_chart(fig, width="stretch")
            else:
                st.dataframe(_b_for_sort.round(4), use_container_width=True)

            # Per-asset breakdown
            _asset_pick = st.selectbox(
                "銘柄を選んでβを棒グラフで確認",
                options=list(_b_for_sort.index),
                index=0,
                key="beta_asset_pick",
            )
            _row = _b_df.loc[_asset_pick].dropna()
            _row = _row.sort_values(key=lambda s: s.abs(), ascending=False)
            _topk = st.slider("表示するETF数（上位|β|）", 3, max(3, min(12, len(_row))), min(8, len(_row)), key="beta_topk")
            _row = _row.iloc[:_topk]

            if go is not None:
                fig2 = go.Figure(data=go.Bar(x=_row.index.tolist(), y=_row.values.tolist()))
                fig2.update_layout(
                    height=plotly_height_px,
                    width=int(display_width_px),
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="ETF",
                    yaxis_title="β",
                    title=f"β breakdown: {_asset_pick}",
                )
                try:
                    st.plotly_chart(fig2, use_container_width=False)
                except TypeError:
                    st.plotly_chart(fig2, width="stretch")
            else:
                st.write(_row)

            with st.expander("βテーブル（数値）", expanded=False):
                st.dataframe(_b_df.round(6), use_container_width=True)
except Exception:
    pass

# =========================
# Excess time series (PF vs Bench)
# =========================
excess_series = None
if (
    bench_index is not None
    and pf_index_common is not None
    and bench_price_common is not None
    and pf_value_common is not None
):
    rel_ratio = (pf_index_common / bench_index).rename("PF / Benchmark")
    rel_diff = (pf_index_common - bench_index).rename("PF Index - Benchmark Index")

    active_log = (np.log(pf_value_common) - np.log(bench_price_common)).rename(
        "Active log (cum)"
    )
    active_log0 = (active_log - active_log.iloc[0]).rename("Active log (cum, 0=start)")
    active_log_index = (100.0 * np.exp(active_log0)).rename("Active (Start=100)")

    excess_series = pd.concat([rel_ratio, rel_diff, active_log_index], axis=1)

if excess_series is not None:
    st.subheader("ベンチマーク超過（時系列）")

    c1, c2 = st.columns(2)

    with c1:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
        plt.plot(excess_series.index, excess_series["PF / Benchmark"].values)
        plt.axhline(1.0, linestyle="--")
        plt.title("Relative Performance: PF / Benchmark (1.0 = same)")
        plt.xlabel("Date")
        plt.ylabel("Ratio")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        st.download_button(
            label="📥 超過（PF÷Benchmark）をPNG保存",
            data=fig_to_png_bytes(fig),
            file_name="excess_pf_div_bench.png",
            mime="image/png",
            key="dl_png_excess_ratio",
        )

    with c2:
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
        plt.plot(
            excess_series.index, excess_series["PF Index - Benchmark Index"].values
        )
        plt.axhline(0.0, linestyle="--")
        plt.title("Index Difference: PF - Benchmark (0 = same)")
        plt.xlabel("Date")
        plt.ylabel("Index diff")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

        st.download_button(
            label="📥 超過（PF−Benchmark）をPNG保存",
            data=fig_to_png_bytes(fig),
            file_name="excess_pf_minus_bench.png",
            mime="image/png",
            key="dl_png_excess_diff",
        )

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    plt.plot(excess_series.index, excess_series["Active (Start=100)"].values)
    plt.axhline(100.0, linestyle="--")
    plt.title("Active Performance (Start=100, >100 = PF better)")
    plt.xlabel("Date")
    plt.ylabel("Active index")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.download_button(
        label="📥 Active（Start=100）をPNG保存",
        data=fig_to_png_bytes(fig),
        file_name="active_start100.png",
        mime="image/png",
        key="dl_png_active",
    )

    with st.expander("超過系列（データ）"):
        st.dataframe(excess_series)
else:
    if benchmark_df is not None:
        st.info(
            "ベンチマークのデータがPF期間に十分揃わないため、超過系列を表示できませんでした。"
        )

# =========================
# Export (PDF / CSV)
# =========================
st.markdown("---")
st.subheader("エクスポート（PDF / CSV）")

export_df = pd.DataFrame(index=pf_value.index)
export_df["PF_VALUE"] = pf_value
export_df["PF_INDEX"] = pf_index

# fixed-weight rebalance
if pv_rt is not None:
    export_df["FIXED_REBAL_VALUE"] = pv_rt.reindex(export_df.index)
if pi_rt is not None:
    export_df["FIXED_REBAL_INDEX"] = pi_rt.reindex(export_df.index)
if tax_paid is not None:
    export_df["FIXED_REBAL_TAX_PAID"] = tax_paid.reindex(export_df.index)
if reb_flag is not None:
    export_df["FIXED_REBAL_FLAG"] = reb_flag.reindex(export_df.index)
if max_rel_dev is not None:
    export_df["FIXED_MAX_REL_WEIGHT_DEV"] = max_rel_dev.reindex(export_df.index)

# static OPT
for label, pv in opt_pf_values.items():
    export_df[f"{label}__VALUE"] = pv.reindex(export_df.index)
for label, pi in opt_pf_indices.items():
    export_df[f"{label}__INDEX"] = pi.reindex(export_df.index)

# walk-forward
if wf_pv is not None:
    export_df["WF_VALUE"] = wf_pv.reindex(export_df.index)
if wf_pi is not None:
    export_df["WF_INDEX"] = wf_pi.reindex(export_df.index)
if wf_tax_paid is not None:
    export_df["WF_TAX_PAID"] = wf_tax_paid.reindex(export_df.index)
if wf_reb_flag is not None:
    export_df["WF_REBALANCE_FLAG"] = wf_reb_flag.reindex(export_df.index)
if wf_max_rel_dev is not None:
    export_df["WF_MAX_REL_WEIGHT_DEV"] = wf_max_rel_dev.reindex(export_df.index)

# benchmark & excess
if bench_index is not None:
    export_df["BENCH_INDEX"] = bench_index.reindex(export_df.index)
if excess_series is not None:
    for c in excess_series.columns:
        export_df[f"EXCESS__{c}"] = excess_series[c].reindex(export_df.index)

st.download_button(
    label="📥 時系列CSV（PF/Bench/最適化/リバランス）をダウンロード",
    data=df_to_csv_bytes(export_df),
    file_name="timeseries_export.csv",
    mime="text/csv",
    key="dl_csv_timeseries",
)

st.download_button(
    label="📥 価格CSV（JPY換算・PF構成銘柄）をダウンロード",
    data=df_to_csv_bytes(prices_jpy_pf_aligned),
    file_name="prices_jpy_pf.csv",
    mime="text/csv",
    key="dl_csv_prices",
)

st.download_button(
    label="📥 指標CSVをダウンロード",
    data=df_to_csv_bytes(dfm),
    file_name="metrics.csv",
    mime="text/csv",
    key="dl_csv_metrics",
)


# =========================
# Export: weights (portfolio allocations)
# =========================
# Build a unified weights table for practical investing reference.
# Rows: instruments, Cols: portfolio labels.
_name_by_ticker = selected_df.set_index("ticker")["display_name"].to_dict()
_row_index = [f"{_name_by_ticker.get(t, t)} ({t})" for t in prices_jpy_pf_aligned.columns]

def _weights_series_to_named(w: pd.Series) -> pd.Series:
    w2 = w.reindex(prices_jpy_pf_aligned.columns).astype(float).fillna(0.0)
    s = (w2 / w2.sum()) if w2.sum() != 0 else w2
    s.index = _row_index
    return s

weights_tables_for_pdf: dict[str, pd.DataFrame] | None = {}
weights_export_cols: dict[str, pd.Series] = {}

# Buy&Hold (user weights)
try:
    weights_export_cols["Buy&Hold"] = _weights_series_to_named(weights)
except Exception:
    pass

# Fixed-weight rebalance uses the same target weights as Buy&Hold
if rebalance_mode != "なし（買いっぱなし）":
    try:
        weights_export_cols["Fixed-Rebal (Target)"] = _weights_series_to_named(weights)
    except Exception:
        pass

# Static OPT weights
for m, w_opt in opt_weights_map.items():
    try:
        weights_export_cols[f"OPT: {m}"] = _weights_series_to_named(w_opt)
    except Exception:
        pass

# Walk-forward: weights vary over time; export the latest target weights if available
if wf_weights_hist is not None and len(wf_weights_hist) > 0:
    try:
        _last_dt, _last_w = wf_weights_hist[-1]
        weights_export_cols[f"{wf_label} (Latest Target)"] = _weights_series_to_named(_last_w)
    except Exception:
        pass

# Compose into a DataFrame (rows=instruments, cols=portfolios)
weights_export_df = None
if len(weights_export_cols) > 0:
    weights_export_df = pd.concat(weights_export_cols, axis=1)
    weights_tables_for_pdf["Portfolio Weights"] = weights_export_df.copy()
    st.download_button(
        label="📥 ポートフォリオウェイトCSV（配分）をダウンロード",
        data=df_to_csv_bytes(weights_export_df),
        file_name="portfolio_weights.csv",
        mime="text/csv",
        key="dl_csv_weights",
    )

# Walk-forward weights history (optional export)
if wf_weights_hist is not None and len(wf_weights_hist) > 0:
    try:
        wf_weights_hist_df = pd.DataFrame(
            [{"date": d, **w.reindex(prices_jpy_pf_aligned.columns).fillna(0.0).to_dict()} for d, w in wf_weights_hist]
        )
        wf_weights_hist_df = wf_weights_hist_df.set_index("date")
        st.download_button(
            label="📥 WFウェイト履歴CSV（リバランス時の目標配分）をダウンロード",
            data=df_to_csv_bytes(wf_weights_hist_df),
            file_name="wf_weights_history.csv",
            mime="text/csv",
            key="dl_csv_wf_weights_hist",
        )
        weights_tables_for_pdf["WF Weights History (last 10)"] = wf_weights_hist_df.tail(10)
    except Exception:
        pass

# If no weights were built, keep PDF export robust
if weights_tables_for_pdf is not None and len(weights_tables_for_pdf) == 0:
    weights_tables_for_pdf = None

benchmark_name_for_pdf = benchmark_name if benchmark_df is not None else None
pdf_bytes = build_export_pdf_bytes(
    prices_jpy_pf_aligned=prices_jpy_pf_aligned,
    pf_index_to_plot=pf_index_to_plot,
    benchmark_name=benchmark_name_for_pdf,
    excess_series=excess_series,
    metrics_df=dfm,
    weights_tables=weights_tables_for_pdf,
    fig_width=pdf_fig_width,
    fig_height=pdf_fig_height,
    fig_dpi=pdf_fig_dpi,
)

st.download_button(
    label="📄 PDF一括保存（全チャート＋指標表）をダウンロード",
    data=pdf_bytes,
    file_name="report_charts.pdf",
    mime="application/pdf",
    key="dl_pdf_report",
)
