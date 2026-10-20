---
name: redip
description: >-
  Re-runs the TradingAI L0 open-dip bounce pipeline on only the tickers from the
  latest /dip output: sync those symbols, scan, omit green starting bars, sort by
  dip depth ascending, and show reclaiming. Use when the user invokes /redip or
  asks to refresh/rescan the previous /dip candidates.
disable-model-invocation: true
---

# /redip — re-scan prior /dip tickers

## Goal

Same task as `/dip`, but only on the tickers outputted by the task `/dip`.

## Ticker set

1. Take tickers from the **most recent `/dip` run in this conversation** — the filtered (non-green) candidate table that was presented to the user.
2. If that table is empty or there is no prior `/dip` in this chat, **stop** and ask the user to run `/dip` first (or paste the ticker list).
3. Do **not** expand to the full watchlist. Do **not** include green-open names that `/dip` omitted, unless the user explicitly adds them.

## Steps

1. **Sync** only those tickers (required first; run from repo root):

```bash
tradingai-sync-intraday --tickers TICKER1 TICKER2 ...
```

Capture the trailing JSON (`synced`, `bars_written`, `universe`, `errors`).

2. **Fail-fast on sync** — stop and do **not** scan if any of these hold:
   - Process crashes / non-zero exit **and** no usable JSON summary
   - `synced <= 0` or `bars_written <= 0`
   - Non-empty `universe` and every ticker failed (`len(errors) >= len(universe)`)

3. **Partial sync OK**: if only a few of the subset fail, print those `errors` and **continue**.

4. **Scan** the same ticker subset (DB bars; no `--live` unless the user asks):

```bash
tradingai-scan-open-dip --tickers TICKER1 TICKER2 ...
```

Optional: `session_day` only if the user asks for a specific day; default today ET.

5. **Filter**: drop rows where `first_bar_green` is true.

6. **Sort**: remaining candidates by `dip_pct` ascending (deepest first). Tie-break with higher `bounce_score`.

7. **Present** the filtered table (alert/scan only — not financial advice):

| Column | Source |
|--------|--------|
| Ticker | `ticker` |
| Dip % | `dip_pct` |
| Score | `bounce_score` |
| Status | `status` |
| Hammer | `trough_hammer` |
| Engulf | `bullish_engulfing` |
| Last | `last_price` |
| Taker@ | `profit_target.profit_taker_price` |
| Trail on@ | `profit_target.trail_activate_price` |
| Trail % | `profit_target.sell_trail_pct` |
| Trail stop | `profit_target.trail_stop_at_activation_price` |

8. **Show all reclaiming**: from the filtered candidates, present every `status == "reclaiming"` row (same columns, `dip_pct` ascending). Note any green-open reclaiming omitted from this subset. Note sync errors tolerated and how many green opens were dropped.

## Notes

- Prefer the latest `/dip` filtered list even if an older `/redip` ran in between, unless the user says to re-use the last `/redip` tickers.
- After `/redip`, the next `/redip` may use either the prior `/dip` list or the latest `/redip` filtered list if the user says “again” / “these” — default to **latest `/dip`** when ambiguous.
- Same candle-tag and fail-fast notes as `/dip` (`first_bar_green` hard omit; `trough_hammer` / `bullish_engulfing` display only).
