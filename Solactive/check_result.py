# check_result.py
import os
import shutil
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Paths & options ----
PRICES_PATH = "data_sources/stock_prices.csv"                 # dd/mm/YYYY
REF_PATH = "data_sources/index_level_results_rounded.csv"     # dd/mm/YYYY
INDEX_PATH = "export.csv"                                     # YYYY-MM-DD
OUT_GIF = "index_plot.gif"
FRAMES_DIR = "_frames"
FPS = 10          # GIF frames per second
STEP = 1          # 1 = every business day; 2 = every 2 days, etc.
DO_ANIMATE = True


# ---------- Shared utils ----------
def load_prices() -> Tuple[pd.DataFrame, List[str]]:
    """Load price table, keep business days, return (df indexed by Date, tickers list)."""
    px = pd.read_csv(PRICES_PATH)
    px["Date"] = pd.to_datetime(px["Date"], dayfirst=True, errors="raise")
    px = px.sort_values("Date")
    px = px[px["Date"].dt.weekday < 5].reset_index(drop=True)
    tickers = [c for c in px.columns if c != "Date"]
    return px.set_index("Date"), tickers


def build_schedule(px: pd.DataFrame, tickers: List[str]) -> Tuple[Dict[pd.Timestamp, Dict[str, float]], List[pd.Timestamp]]:
    """
    For each month, pick top-3 by last business day of previous month and
    apply weights from the first business day of the current month + 1 day.
    Returns: (effective_date -> weights dict, list of rebalance dates).
    """
    all_days = px.index
    months = pd.PeriodIndex(all_days, freq="M").unique().sort_values()
    schedule, rb_dates = {}, []

    for m in months:
        mask_m = (all_days.to_period("M") == m)
        if not mask_m.any():
            continue
        fbd = all_days[mask_m][0]  # first business day in month

        prev_mask = (all_days.to_period("M") == (m - 1))
        if not prev_mask.any():
            continue
        lbd_prev = all_days[prev_mask][-1]  # last business day previous month

        prev_prices = px.loc[lbd_prev, tickers].sort_values(ascending=False)
        top3 = list(prev_prices.index[:3])

        w = {t: 0.0 for t in tickers}
        if len(top3) > 0:
            w[top3[0]] = 0.50
        if len(top3) > 1:
            w[top3[1]] = 0.25
        if len(top3) > 2:
            w[top3[2]] = 0.25

        idx_in = all_days.get_indexer([fbd], method="backfill")[0]
        if idx_in + 1 < len(all_days):
            eff_from = all_days[idx_in + 1]
            schedule[eff_from] = w
            rb_dates.append(eff_from)

    return schedule, sorted(rb_dates)


# ---------- 1) Rebalance calendar (debug print) ----------
def print_schedule() -> None:
    px, tickers = load_prices()
    rng = px.index
    months = pd.PeriodIndex(rng, freq="M").unique().sort_values()
    rows = []

    for m in months:
        fbd = px.index[(px.index.to_period("M") == m)][0]
        prev_mask = (px.index.to_period("M") == (m - 1))
        if not prev_mask.any():
            continue

        lbd_prev = px.index[prev_mask][-1]
        top3 = list(px.loc[lbd_prev, tickers].sort_values(ascending=False).index[:3])

        i = rng.get_indexer([fbd], method="backfill")[0]
        eff_from = rng[i + 1] if i + 1 < len(rng) else None

        # guard in case <3 tickers
        row = {
            "month": str(m),
            "lbd_prev": lbd_prev.date(),
            "fbd": fbd.date(),
            "effective_from": eff_from.date() if eff_from is not None else None,
            "top1": top3[0] if len(top3) > 0 else None,
            "top2": top3[1] if len(top3) > 1 else None,
            "top3": top3[2] if len(top3) > 2 else None,
        }
        rows.append(row)

    diag = pd.DataFrame(rows)
    print("\n=== Rebalance calendar (debug) ===")
    print(diag.to_string(index=False))
    print()


# ---------- 2) Day-by-day comparison & summary ----------
def compare_series() -> None:
    calc = pd.read_csv(INDEX_PATH)
    calc["Date"] = pd.to_datetime(calc["Date"], format="ISO8601", errors="raise")

    ref = pd.read_csv(REF_PATH)
    ref["Date"] = pd.to_datetime(ref["Date"], format="%d/%m/%Y", errors="raise")

    calc_dates = set(calc["Date"])
    ref_dates = set(ref["Date"])
    print(f"Dates in calc: {len(calc_dates)} | Dates in ref: {len(ref_dates)}")

    only_in_calc = sorted(calc_dates - ref_dates)
    only_in_ref = sorted(ref_dates - calc_dates)
    if only_in_calc:
        print(f"Dates only in calc (first 5): {only_in_calc[:5]}")
    if only_in_ref:
        print(f"Dates only in ref  (first 5): {only_in_ref[:5]}")
    print()

    cmp = calc.merge(ref, on="Date", how="inner", suffixes=("_calc", "_ref")).sort_values("Date")
    cmp["calc_round2"] = cmp["index_level_calc"].round(2)
    cmp["abs_diff"] = (cmp["calc_round2"] - cmp["index_level_ref"]).abs()
    cmp["pct_diff"] = (cmp["abs_diff"] / cmp["index_level_ref"]) * 100

    print("=== Day-by-day comparison ===")
    for _, r in cmp.iterrows():
        print(
            f"{r['Date'].date()} | Ref={r['index_level_ref']:.2f} | "
            f"Calc={r['calc_round2']:.2f} | Abs diff={r['abs_diff']:.2f} | "
            f"% diff={r['pct_diff']:.4f}%"
        )

    print("\nOverall:")
    print(f"Compared days: {len(cmp)}")
    print(f"Mean absolute diff: {cmp['abs_diff'].mean():.4f}")
    print(f"Mean % diff: {cmp['pct_diff'].mean():.6f}%")
    print(f"Max % diff: {cmp['pct_diff'].max():.6f}%")


# ---------- 3) Monthly metrics -> metrics.csv ----------
def compute_metrics() -> None:
    START, END = "2020-01-01", "2020-12-31"

    px, tickers = load_prices()
    px = px.loc[START:END, tickers]
    rets = px.pct_change()

    idx = pd.read_csv(INDEX_PATH)
    idx["Date"] = pd.to_datetime(idx["Date"], format="ISO8601", errors="raise")
    idx = idx.set_index("Date").loc[START:END]
    idx = idx.rename(columns={"index_level": "L"})

    common = px.index.intersection(idx.index)
    px, rets, idx = px.loc[common], rets.loc[common], idx.loc[common]

    schedule, _ = build_schedule(px, tickers)
    dates = idx.index

    # Daily target weights (step-wise by schedule)
    W_target = pd.DataFrame(0.0, index=dates, columns=tickers)
    current = {t: 0.0 for t in tickers}
    for d in dates:
        if d in schedule:
            current = schedule[d]
        for t in tickers:
            W_target.at[d, t] = current.get(t, 0.0)

    # Realized weights (drift between rebalances)
    W_real = pd.DataFrame(0.0, index=dates, columns=tickers)
    prev = None
    for i, d in enumerate(dates):
        wt = W_target.loc[d]
        if i == 0 or not wt.equals(W_target.iloc[i - 1]):
            prev = wt.values
        else:
            r = rets.loc[d].fillna(0.0).values
            prev = prev * (1.0 + r)
            s = prev.sum() or 1.0
            prev = prev / s
        W_real.loc[d] = prev

    # Returns & residual
    idx_ret = idx["L"].pct_change().fillna(0.0)
    port_ret = (W_target * rets).sum(axis=1).fillna(0.0)
    residual = idx_ret - port_ret
    abs_res = residual.abs()

    # Concentration
    HHI = (W_target ** 2).sum(axis=1)
    N_eff = 1.0 / HHI.replace(0, np.nan)

    # Drift and turnover
    drift = 0.5 * (W_real.sub(W_target).abs().sum(axis=1))
    turnover = pd.Series(0.0, index=dates)
    prev_tgt = None
    for d in dates:
        if d in schedule:
            if prev_tgt is None:
                turnover.loc[d] = 0.5 * np.abs(
                    np.array([schedule[d].get(t, 0.0) for t in tickers]) - 0
                ).sum()
            else:
                a = np.array([prev_tgt.get(t, 0.0) for t in tickers])
                b = np.array([schedule[d].get(t, 0.0) for t in tickers])
                turnover.loc[d] = 0.5 * np.abs(a - b).sum()
            prev_tgt = schedule[d]

    # Component changes at rebalance points
    comp_changes = pd.Series(0, index=dates, dtype=int)
    prev_names = None
    for d in dates:
        if d in schedule:
            names = tuple([t for t, w in sorted(schedule[d].items()) if w > 0])
            comp_changes.loc[d] = len(names) if prev_names is None else len(
                set(names).symmetric_difference(set(prev_names))
            )
            prev_names = names

    M = pd.DataFrame(
        {
            "idx_ret": idx_ret,
            "port_ret": port_ret,
            "residual": residual,
            "abs_res": abs_res,
            "HHI": HHI,
            "N_eff": N_eff,
            "drift": drift,
            "turnover": turnover,
            "comp_changes": comp_changes,
        }
    )
    M["month"] = M.index.to_period("M")

    monthly = (
        M.groupby("month")
        .agg(
            idx_return=("idx_ret", lambda s: (1 + s).prod() - 1),
            mean_abs_resid=("abs_res", "mean"),
            te_residual=("residual", "std"),
            avg_HHI=("HHI", "mean"),
            avg_Neff=("N_eff", "mean"),
            max_drift=("drift", "max"),
            turnover=("turnover", "sum"),
            constituent_changes=("comp_changes", "sum"),
            days=("idx_ret", "size"),
        )
        .reset_index()
    )

    # Join a small audit table (top-3 and dates)
    audit_rows = []
    px_idx = px.index
    for m in monthly["month"]:
        mask_m = (px_idx.to_period("M") == m)
        if not mask_m.any():
            audit_rows.append(
                {
                    "month": m,
                    "lbd_prev": None,
                    "fbd": None,
                    "effective_from": None,
                    "top1": None,
                    "top2": None,
                    "top3": None,
                }
            )
            continue

        fbd = px_idx[mask_m][0]
        prev_mask = (px_idx.to_period("M") == (m - 1))
        if not prev_mask.any():
            audit_rows.append(
                {
                    "month": m,
                    "lbd_prev": None,
                    "fbd": fbd.date(),
                    "effective_from": None,
                    "top1": None,
                    "top2": None,
                    "top3": None,
                }
            )
            continue

        lbd_prev = px_idx[prev_mask][-1]
        prev_prices = px.loc[lbd_prev].sort_values(ascending=False)
        top3 = list(prev_prices.index[:3])

        i_fbd = px_idx.get_indexer([fbd], method="backfill")[0]
        eff_from = (
            px_idx[i_fbd + 1] if i_fbd + 1 < len(px_idx) else None
        )

        audit_rows.append(
            {
                "month": m,
                "lbd_prev": lbd_prev.date(),
                "fbd": fbd.date(),
                "effective_from": eff_from.date() if eff_from is not None else None,
                "top1": top3[0] if len(top3) > 0 else None,
                "top2": top3[1] if len(top3) > 1 else None,
                "top3": top3[2] if len(top3) > 2 else None,
            }
        )

    audit_df = pd.DataFrame(audit_rows)
    monthly = monthly.merge(audit_df, on="month", how="left")

    cols = [
        "month",
        "lbd_prev",
        "fbd",
        "effective_from",
        "top1",
        "top2",
        "top3",
        "days",
        "idx_return",
        "mean_abs_resid",
        "te_residual",
        "avg_HHI",
        "avg_Neff",
        "max_drift",
        "turnover",
        "constituent_changes",
    ]
    monthly = monthly[cols].sort_values("month")
    monthly.to_csv("metrics.csv", index=False)
    print("Generated metrics.csv (monthly summary).")


# ---------- 4) GIF of index + current constituents (cleans temp PNGs) ----------
def animate_index() -> None:
    print("\n=== Building index GIF ===")
    px, tickers = load_prices()

    idx = pd.read_csv(INDEX_PATH)
    idx["Date"] = pd.to_datetime(idx["Date"], format="ISO8601", errors="raise")
    idx = idx.sort_values("Date").set_index("Date")

    common = px.index.intersection(idx.index)
    px, idx = px.loc[common], idx.loc[common]

    px_norm = px / px.iloc[0] * 100.0
    idx_norm = idx["index_level"] / idx["index_level"].iloc[0] * 100.0

    schedule, rb_dates = build_schedule(px, tickers)

    y_min = min(idx_norm.min(), px_norm.min().min())
    y_max = max(idx_norm.max(), px_norm.max().max())
    pad = (y_max - y_min) * 0.05
    y_min, y_max = y_min - pad, y_max + pad

    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    dates = idx_norm.index[::STEP]
    last_w = {t: 0.0 for t in tickers}

    for i, d in enumerate(dates, 1):
        if d in schedule:
            last_w = schedule[d]
        constituents = [t for t, wt in sorted(last_w.items()) if wt > 0]

        sub_idx = idx_norm.loc[idx_norm.index[0] : d]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sub_idx.index, sub_idx.values, label="Index (rebased=100)", linewidth=2.0)

        for t in constituents:
            ser = px_norm[t].loc[px_norm.index[0] : d]
            ax.plot(ser.index, ser.values, label=t, linewidth=1.5)

        for rb in rb_dates:
            if rb <= d:
                ax.axvline(rb, linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(f"Index vs current constituents — {d.date()}")
        ax.set_xlim(idx_norm.index[0], idx_norm.index[-1])
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Date")
        ax.set_ylabel("Rebased to 100 at start")
        ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)

        weights_txt = "Weights: " + ", ".join(
            f"{t}={last_w.get(t, 0) * 100:.0f}%" for t in constituents
        )
        ax.text(0.01, 0.02, weights_txt, transform=ax.transAxes)

        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:04d}.png")
        fig.tight_layout()
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)

    frames = [
        imageio.imread(os.path.join(FRAMES_DIR, f))
        for f in sorted(os.listdir(FRAMES_DIR))
        if f.endswith(".png")
    ]
    imageio.mimsave(OUT_GIF, frames, fps=FPS)
    print(f"GIF saved: {OUT_GIF}")

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)


# ---------- Main ----------
if __name__ == "__main__":
    compare_series()
    print_schedule()
    compute_metrics()
    if DO_ANIMATE:
        animate_index()


