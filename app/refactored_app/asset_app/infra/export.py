# -*- coding: utf-8 -*-
"""Export helpers (PDF/CSV)."""

from __future__ import annotations

from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# Export helpers
# =========================
def fig_to_png_bytes(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


def series_to_csv_bytes(s: pd.Series, name: str = "value") -> bytes:
    return df_to_csv_bytes(s.to_frame(name=name))


def add_df_as_table_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    title: str,
    fig_width: int,
    fig_height: int,
    fig_dpi: int,
):
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, loc="left")

    df2 = df.copy()
    for c in df2.columns:
        try:
            df2[c] = pd.to_numeric(df2[c])
            df2[c] = df2[c].map(lambda x: f"{x:.6f}")
        except Exception:
            df2[c] = df2[c].astype(str)

    cell_text = df2.reset_index().values.tolist()
    col_labels = df2.reset_index().columns.tolist()

    table = ax.table(
        cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_export_pdf_bytes(
    prices_jpy_pf_aligned: pd.DataFrame,
    pf_index_to_plot: dict[str, pd.Series],
    benchmark_name: str | None,
    excess_series: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    weights_tables: dict[str, pd.DataFrame] | None,
    fig_width: int,
    fig_height: int,
    fig_dpi: int,
) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # 1) 個別銘柄（JPY換算）
        for tkr in prices_jpy_pf_aligned.columns:
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
            plt.plot(prices_jpy_pf_aligned.index, prices_jpy_pf_aligned[tkr].values)
            plt.title(f"{tkr} (JPY)")
            plt.xlabel("Date")
            plt.ylabel("Price (JPY)")
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) PF指数比較（Start=100）
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
        for label, s in pf_index_to_plot.items():
            plt.plot(s.index, s.values, label=label)
        plt.title(
            "PF Index (Start=100)" + (f" vs {benchmark_name}" if benchmark_name else "")
        )
        plt.xlabel("Date")
        plt.ylabel("Index")
        plt.xticks(rotation=30)
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3) 超過系列
        if excess_series is not None:
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
            plt.plot(excess_series.index, excess_series["PF / Benchmark"].values)
            plt.axhline(1.0, linestyle="--")
            plt.title("PF / Benchmark (1.0 = same)")
            plt.xlabel("Date")
            plt.ylabel("Ratio")
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
            plt.plot(
                excess_series.index, excess_series["PF Index - Benchmark Index"].values
            )
            plt.axhline(0.0, linestyle="--")
            plt.title("PF Index - Benchmark Index (0 = same)")
            plt.xlabel("Date")
            plt.ylabel("Index diff")
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(fig_width, fig_height), dpi=fig_dpi)
            plt.plot(excess_series.index, excess_series["Active (Start=100)"].values)
            plt.axhline(100.0, linestyle="--")
            plt.title("Active (Start=100, >100 = PF better)")
            plt.xlabel("Date")
            plt.ylabel("Active index")
            plt.xticks(rotation=30)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 4) 指標表
        if metrics_df is not None and not metrics_df.empty:
            add_df_as_table_page(
                pdf, metrics_df, "Metrics", fig_width, fig_height, fig_dpi
            )


        # 5) ウェイト表（ポートフォリオ配分）
        if weights_tables is not None:
            for title, wdf in weights_tables.items():
                try:
                    if wdf is None:
                        continue
                    if isinstance(wdf, pd.Series):
                        wdf2 = wdf.to_frame("weight")
                    else:
                        wdf2 = wdf.copy()
                    if wdf2.empty:
                        continue
                    add_df_as_table_page(
                        pdf, wdf2, f"Weights - {title}", fig_width, fig_height, fig_dpi
                    )
                except Exception:
                    # export should be best-effort; avoid failing the whole PDF
                    pass

    buf.seek(0)
    return buf.getvalue()
