import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class IndexModel:
    def __init__(self) -> None:
        self._prices_path = Path("data_sources/stock_prices.csv")
        self._out: Optional[pd.DataFrame] = None

        # Rule 1 (Total Return): expects TR prices; if not, dividends must be added upstream.
        # Rule 2 (Universe A..J): assumes file columns match the universe.
        # load prices (dd/mm/yyyy)
        df = pd.read_csv(self._prices_path)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values("Date", ignore_index=True)
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)

        self.prices = df
        self.tickers: List[str] = [c for c in df.columns if c != "Date"]
        self.returns = df.set_index("Date")[self.tickers].pct_change()

    def calc_index_level(self, start_date: dt.date, end_date: dt.date) -> None:
        # Rule 7 (Start date 2020-01-01): provided by caller via start_date.
        px = self.prices.set_index("Date").loc[str(start_date):str(end_date)].copy()
        rets = self.returns.loc[px.index].copy()
         # Rule 8 (Biz days Mon–Fri, no extra holidays): filter weekdays only.
         
        idx = pd.Series(index=px.index, dtype=float)
        idx.iloc[0] = 100.0  # Rule 6: index starts at 100.

        # Rule 5: selection becomes effective COB on FBD -> applied from next business day.
        eff_weights = self._build_effective_weight_schedule(px.index)

        current_w: Dict[str, float] = {}
        for i in range(1, len(idx)):
            d = idx.index[i]
            if d in eff_weights:
                current_w = eff_weights[d]  # Rule 4: weights are 50/25/25 (set in helper).

            # Rule 1: total return via returns; ensure inputs include dividends if required.
            day_ret = 0.0
            if current_w:
                r_t = rets.loc[d]
                day_ret = float(sum(r_t[t] * current_w.get(t, 0.0) for t in self.tickers))

            idx.iloc[i] = idx.iloc[i - 1] * (1.0 + day_ret)

        self._out = pd.DataFrame({"Date": idx.index, "index_level": idx.values})

    def export_values(self, file_name: str) -> None:
        if self._out is None:
            raise RuntimeError("Call calc_index_level(...) first")
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        self._out.to_csv(file_name, index=False)

    # helpers
    def _build_effective_weight_schedule(
        self, index_dates: pd.DatetimeIndex
    ) -> Dict[pd.Timestamp, Dict[str, float]]:
        px = self.prices.set_index("Date")
        all_days = px.index

        months = pd.PeriodIndex(index_dates, freq="M").unique().sort_values()
        eff: Dict[pd.Timestamp, Dict[str, float]] = {}

        for m in months:
            month_mask = all_days.to_period("M") == m
            if not month_mask.any():
                continue
            fbd = all_days[month_mask][0]  # FBD of month

            prev_mask = all_days.to_period("M") == (m - 1)
            if not prev_mask.any():
                continue
            lbd_prev = all_days[prev_mask][-1]  # LBD of prior month

            # Rule 3: select top-3 by market cap on LBD (here proxied by price column ordering).
            prev_prices = px.loc[lbd_prev, self.tickers].sort_values(ascending=False)
            top3 = list(prev_prices.index[:3])

            # Rule 4: 50/25/25 weights.
            w = {t: 0.0 for t in self.tickers}
            if len(top3) >= 1:
                w[top3[0]] = 0.50
            if len(top3) >= 2:
                w[top3[1]] = 0.25
            if len(top3) >= 3:
                w[top3[2]] = 0.25

            # Rule 5: effective from the day after FBD (to reflect COB on FBD).
            i_after = index_dates.get_indexer([fbd], method="backfill")[0] + 1
            if i_after < len(index_dates):
                eff[index_dates[i_after]] = w

        return eff




