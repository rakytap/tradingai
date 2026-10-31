---
name: buy-trail
description: >-
  Places an IBKR budget LIMIT buy at mid (no STP), then background-monitors
  fills and attaches a 0.6% TRAIL LIMIT from the activate mid; tightens to
  0.05%/0.03%/0.01% at 1×/1.5×/2× fee-aware target return (5s poll after
  first tighten); exits when position is flat. Use when the user invokes
  /buy-trail or asks for buy-with-trailing-stop.
disable-model-invocation: true
---

# /buy-trail — budget buy + stepped protective trail

## Goal

1. Place a **LIMIT** buy sized like `/buy` (full budget, integer shares).
2. Default **limit = current IB mid** (also the **trail activate** price).
3. **Do not** attach STP — use a trailing sell instead.
4. After submit, a **background** job:
   - Polls every **10s** until the first tighten, then every **5s**.
   - Waits until the buy is **fully filled**.
   - Places **TRAIL LIMIT** sell: trail **0.6%**, stop ≈ `activate_mid * (1 - 0.006)`.
   - When mid ≥ fee-aware **1×** `TRADINGAI_OPEN_DIP_TARGET_RETURN` → trail **0.05%**.
   - When mid ≥ fee-aware **1.5×** target → trail **0.03%**.
   - When mid ≥ fee-aware **2×** target → trail **0.01%**.
   - When the **position is flat** (shares sold) → **terminate** the job.

## Defaults

| Input | Default |
|-------|---------|
| Buy | LMT @ mid, spend **100%** of budget, **no STP** |
| Activate | buy mid / limit |
| Initial trail | **0.6%** (IB percent points) |
| @ 1× target | trail **0.05%**, poll → **5s** |
| @ 1.5× target | trail **0.03%** |
| @ 2× target | trail **0.01%** |
| Target | `TRADINGAI_OPEN_DIP_TARGET_RETURN` (fee-aware) |
| Poll | **10s** then **5s** after first tighten |
| `submit` | **false** (dry-run buy only; no watcher) |

## Steps

1. Resolve buy args (`symbol`, budget, price, …) and trail args
   (`initial-trail-percent`, `tight-trail-percent`, `trail-1-5x-percent`,
   `trail-2x-percent`, `target-return`).
2. Prefer MCP `tradingai-ibkr` → `buy_trail_start`, or CLI:

```bash
# Dry-run buy + trail plan
tradingai-buy-trail COIN --budget 6000

# Live buy + background trail watcher
tradingai-buy-trail COIN --budget 6000 --submit --yes

# Custom 1× profit level that triggers first tighten
tradingai-buy-trail COIN --budget 6000 --target-return 0.01 --submit --yes

tradingai-buy-trail --status
tradingai-buy-trail --stop <job_id>
```

3. Present `buy.order_id`, `job_id`, `pid`, log path, activate/trail plan.
4. Live needs `TRADINGAI_IBKR_ALLOW_ORDERS=1`.

## Notes

- State under `data/buy_trail/<job_id>/` (or `TRADINGAI_BUY_TRAIL_DIR`).
- Distinct from `/autosell` (symbol-wide unpaired buys) and plain `/buy` (STP child).
- Restart `tradingai-ibkr` MCP after code/env changes.
