---
name: dipwalk
description: >-
  Walks the equity universe in randomized order (each ticker once): sync and
  scan one stock at a time, keep green-open names, and export every name with
  dip deeper than 4% (reclaiming, stabilizing, or plunging) to a Markdown file.
  Use when the user invokes /dipwalk or asks for a randomized one-by-one dip walk
  with MD export.
disable-model-invocation: true
---

# /dipwalk — randomized one-by-one open-dip walk

## Goal

Traverse the stock universe in a **randomized order**, visiting each equity
**once per run**. For each ticker: sync intraday bars, then analyze the open-dip
bounce. **Export incrementally during the run**: as soon as a ticker finishes
analysis and meets the criteria, append it to the Markdown file (do **not**
buffer hits until the end). Include any stock whose dip is **larger than 4%**
(`dip_pct <= -4.0`), regardless of status (`reclaiming`, `stabilizing`, or
`plunging` / falling).

## Relation to `/dipbatch`

Same data path (IBKR sync → `tradingai-scan-open-dip` / `evaluate_ticker`) and
same green-open policy as `/dipbatch` (**keep** `first_bar_green`; mark via
Green column). Differences:

| | `/dipbatch` | `/dipwalk` |
|---|-------------|------------|
| Order | Sorted batches of 50 | **Shuffled** full equity list |
| Unit of work | Batch sync + bullish subset scan | **One ticker** sync + scan |
| Sector filter | Bullish-sector only for scan | **Full** non-ETF sync universe |
| Output | Chat tables per batch + summary | **Live-updated Markdown file** + brief chat summary |
| Export timing | After each batch | **Immediately after each qualifying ticker** (append+flush) |
| Export filter | All candidates (any dip ≥ threshold in scan) | Only rows with **`dip_pct <= -4.0`**, all statuses |

## Steps

1. **Resolve universe** (repo root):

   ```bash
   python -c "from tradingai.data.config import get_sync_tickers; print('\n'.join(get_sync_tickers()))"
   ```

   Fall back to `list_tickers()` if empty. Drop ETFs (`is_etf`). Deduplicate,
   then **shuffle with a fixed seed for reproducibility within the run**:

   ```python
   import random
   rng = random.Random()  # or Random(seed) if user gave a seed
   rng.shuffle(equities)
   ```

   Persist the shuffled order to a temp file (e.g. `/tmp/dipwalk_order.txt`) so
   the run can resume after a mid-walk failure without reshuffling.

2. **Choose export path** (create parent dirs if needed):

   - Default: `data/dipwalk/dipwalk_YYYY-MM-DD_HHMM_ET.md` (session day + time in America/New_York)
   - Or a path the user specifies

   **Create the Markdown file before the walk starts.** Write the header and an
   empty hits table (column header row) immediately. Keep the file open for
   append (or reopen in append mode) for the whole run.

3. **For each ticker `T` in shuffled order** (1-based index `i/N`):

   **3a. Sync this ticker only:**

   ```bash
   tradingai-sync-intraday --tickers T
   ```

   Capture trailing JSON (`synced`, `bars_written`, `errors`).

   **Fail-fast on this ticker only if sync is wholesale-broken for that call:**
   - Process crash / non-zero exit **and** no usable JSON → **stop the run**
     (TWS likely down); keep the MD file with whatever was already exported
   - `synced <= 0` or `bars_written <= 0` → note skip, **continue** to next ticker
   - Per-ticker qualify/errors in JSON → note and **continue**

   **3b. Scan** (DB bars; no `--live` unless user asks):

   ```bash
   tradingai-scan-open-dip --tickers T
   ```

   Or `evaluate_ticker(T, prefer_db=True)`. Collect the row from `candidates`
   **or** `plunging` (falling) if present.

   **3c. Export immediately if criteria met (required — during the run):**
   - After this ticker’s analysis finishes, if a row exists and
     `dip_pct <= -4.0`, **append that row to the Markdown hits table right now**
     (before syncing/scanning the next ticker).
   - `flush` / fsync so the file on disk is readable mid-run.
   - Include reclaiming, stabilizing, **and** plunging. Keep green-open names.
   - Do **not** accumulate hits only in memory for a single end-of-run write.
   - Optionally print a one-line chat note for the hit.

   **3d. Progress:** every 25 tickers (or on each export hit), print a short
   progress line: `i/N visited, hits=H, last_hit=TICKER`.

4. **Finalize the Markdown file** (after all tickers) with a summary section
   appended (hits are already in the file from step 3c):

   - Tickers visited / skipped / sync errors
   - Hit count (`dip_pct <= -4`)
   - Counts by status (reclaiming / stabilizing / plunging)
   - Green-open count among hits

5. **Chat summary:** path to the MD file, hit count, deepest few tickers
   (`dip_pct` ascending), and note that green opens were kept.

## Markdown table columns (append order = discovery / export order; then a
final sorted section by `dip_pct` ascending)

| Column | Source |
|--------|--------|
| Ticker | `ticker` |
| Dip % | `dip_pct` |
| Score | `bounce_score` |
| Status | `status` (`reclaiming` / `stabilizing` / `plunging`) |
| Green | `first_bar_green` |
| Hammer | `trough_hammer` |
| Engulf | `bullish_engulfing` |
| Last | `last_price` |
| Taker@ | `profit_target.profit_taker_price` |
| Trail on@ | `profit_target.trail_activate_price` |
| Trail % | `profit_target.sell_trail_pct` |
| Trail stop | `profit_target.trail_stop_at_activation_price` |

Example row block to append:

```markdown
| TICKER | -5.12 | 0.42 | reclaiming | Y |  |  | 12.34 | ... |
```

After the walk, append a **Sorted by dip depth** section with the same hits
ordered by `dip_pct` ascending (the live table above stays in discovery order).

## Notes

- **Incremental export is mandatory:** write each qualifying hit to the MD file
  immediately after that stock is analyzed; the file must grow during the run.
- Dip threshold is **hard**: only export if `dip_pct <= -4.0` (more than 4% below prior close). Do not export milder dips.
- `plunging` counts as falling — include it when the dip threshold is met.
- Full-universe one-by-one sync is slow (~1000 IBKR requests). Prefer leaving it running; resume from `/tmp/dipwalk_order.txt` + a progress index if interrupted (hits already on disk remain valid).
- `tradingai-sync-intraday` often exits 0 with per-ticker errors — parse JSON for decisions.
- Alert/scan only — not financial advice.
