---
name: dipbatch
description: >-
  Runs the TradingAI L0 open-dip bounce pipeline in batches of 50: sync
  intraday bars per batch, analyze only bullish-sector names from that batch,
  keep green-open names (mark them; do not drop), print intermediate results
  after each batch, then a combined summary. Use when the user invokes
  /dipbatch or asks to batch-sync and scan open dips.
disable-model-invocation: true
---

# /dipbatch — batched open-dip bounce scan

## Goal (verbatim)

Synchronize the universe of the stocks in batches of 50 assets. Then analyze those stocks from this batch which are in the bullish sectors. Print the intermediate results.

## Relation to `/dip`

Same sort/present rules as `.cursor/skills/dip/SKILL.md` (sort by `dip_pct` ascending; show reclaiming; fail-fast only on wholesale sync failure), **except**: do **not** drop green-open (`first_bar_green`) names — keep them in the tables and mark them (column `Green`). Difference vs a full `/dip`: work the equity universe in **batches of 50**, sync each batch, scan only the **bullish-sector** subset of that batch, and **print results after every batch**.

## Steps

1. **Resolve universe** (repo root):

   - Prefer packaged / configured sync universe via Python:
     ```bash
     python -c "from tradingai.data.config import get_sync_tickers; print('\n'.join(get_sync_tickers()))"
     ```
   - Fall back to `list_tickers()` from the DB if the sync list is empty.
   - Drop known ETFs (`is_etf`) so batches are equities-only.
   - Chunk into consecutive batches of **50** (last batch may be smaller).

2. **Resolve L4 bullish sectors once** (for the whole run):

   ```bash
   python -c "from tradingai.analysis.ml.ranking import bullish_sectors_from_regime; print(bullish_sectors_from_regime())"
   ```

   Build the set of registered non-ETF tickers in those sectors (same idea as `bullish_sector_watchlist` without applying the global cap to the full run). If the bullish board is empty, treat all non-ETF universe names as eligible (same fallback as watchlist).

3. **For each batch `i` of 50** (1-based), do the following in order:

   **3a. Sync this batch only** (required; wait for completion):

   ```bash
   tradingai-sync-intraday --tickers T1 T2 ... T50
   ```

   Capture the trailing JSON (`synced`, `bars_written`, `universe`, `errors`).

   **Fail-fast on this batch** — stop the whole `/dipbatch` run (do not continue to later batches) if:
   - Process crashes / non-zero exit **and** no usable JSON summary
   - `synced <= 0` or `bars_written <= 0`
   - Non-empty batch universe and every ticker failed (`len(errors) >= len(universe)`)

   **Partial sync OK**: if only a few tickers in the batch fail, note those `errors` and continue.

   Skip sync for a batch only if the user explicitly says not to fetch / re-sync.

   **3b. Bullish filter for this batch**: `scan_tickers = [t for t in batch if t in bullish_set]`.  
   If `scan_tickers` is empty, print a one-line intermediate note (`Batch i/N: 0 bullish-sector names — skip scan`) and continue to the next batch.

   **3c. Scan** (DB bars; do not pass `--live` unless the user asks):

   ```bash
   tradingai-scan-open-dip --tickers ...scan_tickers...
   ```

   Or MCP `scan_open_dip_bounces` with the same `tickers`.  
   Optional: `session_day` / MCP `session_day` only if the user asks for a specific day; default today ET.

   **Important:** when `--tickers` is passed, the CLI does **not** re-apply the L4 bullish gate — that is why step 3b must filter first.

   **3d. Sort** (do **not** drop green-open):
   - Keep all candidate rows, including `first_bar_green == true`
   - Sort by `dip_pct` ascending; tie-break higher `bounce_score`

   **3e. Print intermediate results** for this batch (alert/scan only — not financial advice):

   - Header: `### Batch i/N` (tickers synced, bullish scanned, green-open count, sync errors tolerated)
   - Candidate table:

   | Column | Source |
   |--------|--------|
   | Ticker | `ticker` |
   | Dip % | `dip_pct` |
   | Score | `bounce_score` |
   | Status | `status` |
   | Green | `first_bar_green` (mark if true) |
   | Hammer | `trough_hammer` |
   | Engulf | `bullish_engulfing` |
   | Last | `last_price` |
   | Taker@ | `profit_target.profit_taker_price` |
   | Trail on@ | `profit_target.trail_activate_price` |
   | Trail % | `profit_target.sell_trail_pct` |
   | Trail stop | `profit_target.trail_stop_at_activation_price` |

   - **All reclaiming** from this batch’s candidates (same columns; include green-open reclaimings)
   - Optionally note how many candidates had green opens (informational only — not omitted)

   Accumulate all candidates across batches for the final summary.

4. **Final summary** after all batches:

   - Merge accumulated candidates; de-dupe by ticker (keep best / last scan)
   - Re-sort by `dip_pct` ascending
   - Present one combined candidates table + one combined reclaiming table (both keep green-open)
   - Totals: batches run, tickers synced, bullish scanned, green-open count, sync errors tolerated

## Notes

- Batch size is **50** unless the user asks for a different size.
- `first_bar_green`: opening N-minute composite bar green (`DEFAULT_FIRST_BAR_MINUTES`). **`/dipbatch` keeps these names** (unlike `/dip`, which hard-omits them); show via the Green column. Candle tags are ranking/display only.
- Each batch sync is much shorter than a full-universe sync; full `/dipbatch` over ~1000 names is still many IBKR requests overall.
- `tradingai-sync-intraday` often exits 0 even with per-ticker errors — always parse the JSON summary for fail-fast.
- Prefer printing each batch’s tables as soon as that batch finishes (do not wait until the end to show intermediates).
