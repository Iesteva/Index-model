# Assessment Index Modelling — Code Explanation

This project computes the **daily levels of a simple equity index** following the rules from the assessment.
The core logic lives in `index_model/index.py` inside the `IndexModel` class.

## Relevant files

- `data_sources/stock_prices.csv` — daily **total return** prices for `Stock_A` … `Stock_J` (dates in `dd/mm/YYYY`).
- `data_sources/index_level_results_rounded.csv` — **rounded** reference levels (for verification only).
- `index_model/index.py` — index model implementation.
- `__main__.py` — runs the model and writes `export.csv`.
- (Optional) `check_result.py` — utilities to compare with the reference, produce monthly metrics, and a GIF.

## Rules implemented

1. **Universe**: `Stock_A` … `Stock_J`.
2. **Monthly selection (reconstitution)**:
   - On the **last business day of the previous month**, rank constituents by market cap.  
     Because all companies have the **same number of shares**, we use **price** as a proxy for market cap.
   - Pick the **Top 3**.
3. **Weights**: 50% (rank 1), 25% (rank 2), 25% (rank 3).
4. **Effective time**: “effective **at the close** of the first business day of the month” ⇒ the new weights **apply from the next business day**.
5. **Business days**: Monday–Friday (no additional holidays).
6. **Start level**: 100 on **2020-01-01**.
7. **Daily index evolution**

$$
L_t = L_{t-1}\left(1 + \sum_{i} w_{i,t}\, r_{i,t}\right)
$$

where $r_{i,t}$ are daily returns computed from total-return prices.

## How the code works

### 1) Data loading and preparation (`IndexModel.__init__`)
- Read `stock_prices.csv`, parse `Date` with `dayfirst=True`.
- Sort by date and keep **Monday–Friday** only.
- Store:
  - `self.prices`: DataFrame with dates and prices.
  - `self.tickers`: list of stock columns.
  - `self.returns`: daily returns (via `pct_change()`), indexed by **date**.

### 2) Effective weight schedule (`_build_effective_weight_schedule`)
For each month present in the requested date range:
- Identify **FBD** (first business day of the month) and **LBD_prev** (last business day of the previous month).
- On `LBD_prev`, rank prices and select the **Top 3** constituents.
- Build the target weight vector `{50%, 25%, 25%}`.
- Mark these weights as **effective from the business day *after* the FBD** (because the rule says “effective at close” of the FBD).
- Return a dict:  
  `effective_weights[effective_date] = {ticker: weight, ...}`.

### 3) Index level calculation (`calc_index_level(start_date, end_date)`)

- Slice the data to `[start_date, end_date]`.
- Initialize the index series: `idx.iloc[0] = 100`.
- Build the effective weight schedule (step 2).
- Keep `current_w` as the **active weights**:
  - If the current date is an effective date, update `current_w`.
  - Otherwise keep yesterday’s weights.
- For each day $t>t_0$:
  - Take the vector of daily returns $r_t$.
  - Compute portfolio return $\sum_{i} w_{i,t}\, r_{i,t}$.
  - - **Update (plain code):** `L_t = L_{t-1} * (1 + portfolio_return_t)`
- Store the result as a DataFrame with `Date, index_level` (no rounding) in `self._out`.



### 4) Export (`export_values(file_name)`)
Write `self._out` to CSV (`export.csv`).

## Assumptions & notes

- **Market cap ≈ price** because all companies have the same number of shares.
- **Effective t+1**: weights become active **the day after** the FBD (rule says “effective at close”).
- **Business days** only (Mon–Fri); no extra holidays.
- The reference CSV is **rounded to 2 decimals**; the model produces **high-precision** levels, so tiny differences against the rounded file are expected.

## How to run

```bash
cd Solactive
pip install -r requirements.txt
python __main__.py          # generates export.csv with Date, index_level

