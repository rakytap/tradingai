---
name: autosell
description: >-
  Starts a background IBKR autosell watcher that polls TWS fills every 15s and
  places tradingai-sell-style TRAIL LIMIT sells for unpaired buys. Use when the
  user invokes /autosell or asks to auto-sell after buys.
disable-model-invocation: true
---

# /autosell — IBKR unpaired-buy trail sell watcher

## Goal

Run a **background** job that monitors realized IBKR buy fills and, when a buy
has no FIFO sell pair, issues a TRAIL LIMIT sell for **that buy’s share count**
using the same arguments as `/sell-trail` / `tradingai-sell-trail` /
MCP `sell_stock_trail` (not plain `/sell`, which is LIMIT-only).

## Defaults

| Input | Default |
|-------|---------|
| poll interval | **15s** |
| sell order | TRAIL LIMIT (same as `/sell-trail`) |
| sell qty | exact unpaired buy shares (optional `--qty` = max cap) |
| `target_return` / `price` / `trail_percent` | same as `tradingai-sell-trail` |
| `submit` | **false** (dry-run sells) unless user asks live |

## Steps

1. Resolve trail-sell args (`symbol` required; optional price / target-return / trail / account / submit).
2. Prefer MCP `tradingai-ibkr` → `autosell_watch_start`. Or CLI:

```bash
# Background dry-run watcher
tradingai-autosell COIN

# Live autosells (ALLOW_ORDERS=1); confirm unless --yes
tradingai-autosell COIN --target-return 0.01 --submit --yes

# Status / stop
tradingai-autosell --status
tradingai-autosell --status COIN_20260101T120000Z
tradingai-autosell --stop COIN_20260101T120000Z
```

3. Present `job_id`, `pid`, `log_path`, and remind that live mode needs
   `TRADINGAI_IBKR_ALLOW_ORDERS=1`.
4. On follow-up, use `autosell_watch_status` / `autosell_watch_stop`.

## Notes

- Pairing is FIFO on session executions from TWS (`reqExecutions`).
- State/logs under `data/autosell/<job_id>/` (or `TRADINGAI_AUTOSELL_DIR`).
- Restart `tradingai-ibkr` MCP after code/env changes.
