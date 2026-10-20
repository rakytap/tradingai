---
name: dip
description: >-
  Runs the TradingAI L0 open-dip bounce pipeline: sync intraday 1m bars, scan
  dip bounces, omit green starting bars, sort by dip depth ascending, show all
  reclaiming names, and abort only on wholesale sync failure. Use when the user
  invokes /dip or asks to scan open dips, predict dip bounces, or run
  tradingai-scan-open-dip / tradingai-sync-intraday.
disable-model-invocation: true
---

# /dip — open-dip bounce scan

## Goal (verbatim)

Use tradingai-scan-open-dip to predict dip bounces. Before prediction use tradingai-sync-intraday to update time series. From the results omit stocks with green strating bars. Sort the serults according to the dip depth, in ascending order. If syncronization of stock time series fails, then exit from the process. Dont exit the syncronization process if only few of the stocks fail to sync.

## Steps

1. **Sync** (required first; run from repo root, wait for completion):

```bash
tradingai-sync-intraday
```

Capture the trailing JSON object (`synced`, `bars_written`, `universe`, `errors`).

2. **Fail-fast on sync** — stop and do **not** scan if any of these hold:
   - Process crashes / non-zero exit **and** no usable JSON summary
   - `synced <= 0` or `bars_written <= 0`
   - Non-empty `universe` and every ticker failed (`len(errors) >= len(universe)`)

3. **Partial sync OK**: if only a few tickers fail (e.g. one delisted symbol), print those `errors` and **continue** to the scan.

4. **Scan** (DB bars after sync; do not pass `--live` unless the user asks):

```bash
tradingai-scan-open-dip
```

Optional: `session_day` / MCP `scan_open_dip_bounces` only if the user asks for a specific day; default is today ET. Still sync with the CLI first.

5. **Filter**: drop rows where `first_bar_green` is true (green starting bars). Apply to `candidates` (and mention plunging only if useful).

6. **Sort**: remaining candidates by `dip_pct` ascending (most negative / deepest first). Tie-break with higher `bounce_score` if needed.

7. **Present** a concise table of the filtered candidates (alert/scan only — not financial advice):

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

8. **Show all reclaiming**: from the filtered (non-green) candidates, also present **every** row with `status == "reclaiming"`, same columns, sorted by `dip_pct` ascending. If the main table already includes them, still call out this reclaiming subset (or present it as a second short table). If a reclaiming name was omitted only because of green open, list its ticker under a brief “green-open reclaiming omitted” note.

Also note how many green-open names were omitted and any sync errors that were tolerated.

## Notes

- `first_bar_green`: opening N-minute composite bar green (`DEFAULT_FIRST_BAR_MINUTES` in `src/tradingai/intraday/config.py`). Hard omit stays; candle tags (`trough_hammer`, `bullish_engulfing`) are ranking/display only.
- Sync can take ~1+ minute for the full bullish-sector watchlist.
- `tradingai-sync-intraday` often exits 0 even with per-ticker errors — always parse the JSON summary for fail-fast.
