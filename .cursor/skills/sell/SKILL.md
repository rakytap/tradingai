---
name: sell
description: >-
  Submits an IBKR TWS TRAIL LIMIT sell (default: full open position, fee-aware
  limit at TRADINGAI_OPEN_DIP_TARGET_RETURN on last buy, 0.05% trail) via
  tradingai-ibkr sell_stock_trail or tradingai-sell CLI. Use when the user
  invokes /sell or asks to trail-limit sell a ticker.
disable-model-invocation: true
---

# /sell — IBKR TWS trail-limit sell

## Goal

Submit a sell transaction, trail limit by default. The arguments should be the
stock ticker, amount of stocks (by default all open positions), the limit price
(by default TRADINGAI_OPEN_DIP_TARGET_RETURN profit upon the last buy price of
the stock), and with the trailing amount set to 0.05% by default.

## Defaults

| Input | Default |
|-------|---------|
| `order_type` | `TRAIL LIMIT` |
| `quantity` | full open long position |
| `limit_price` | fee-aware sell at `target_return` on last buy / avg cost |
| `target_return` | `TRADINGAI_OPEN_DIP_TARGET_RETURN` / ≈**0.3836%** net after fees |
| `trail_percent` | **0.05** (IB percent points → **0.05%**) |
| `dry_run` | **true** unless user explicitly asks to submit live |

## Steps

1. **Resolve inputs** from the user message:
   - `symbol` (required)
   - `quantity` (optional → all open)
   - `limit_price` (optional → fee-aware target)
   - `target_return` (optional → env/default ≈0.3836%; fraction, e.g. `0.01` = 1%)
   - `trail_percent` (optional → 0.05)
   - `dry_run` (optional → true)

2. **Prefer MCP** `tradingai-ibkr` → `sell_stock_trail`. Or CLI:

```bash
# Dry-run: full position, fee-aware target limit, 0.05% trail
tradingai-sell COIN

# Custom net profit % (fraction): 1% after fees
tradingai-sell COIN --target-return 0.01

# Explicit size / limit / trail (skips fee-aware target_return math)
tradingai-sell COIN --qty 33 --price 170.65 --trail-percent 0.05

# Live (needs TRADINGAI_IBKR_ALLOW_ORDERS=1)
tradingai-sell COIN --submit
tradingai-sell COIN --target-return 0.01 --submit --yes
```

3. **Fail-fast** on `error`, `status == "rejected"`, or missing position when
   qty omitted.

4. **Confirm before live** — print sizing and require confirmation unless the
   user already asked to submit / passed `--yes`.

5. **Present** a short result table (informational — not financial advice):

| Field | Source |
|-------|--------|
| Symbol | `symbol` |
| Side / type | sell / TRAIL LIMIT |
| Qty | `quantity` |
| Limit / stop ref | `limit_price` / `trail_stop_price` |
| Target return | `sizing.target_return` (when fee-aware) |
| Trail % | `trail_percent` |
| Buy ref | `sizing.buy_price` + source |
| Dry run | `dry_run` |
| Status | `status` / `order_id` |

## Notes

- `target_return` is a **fraction** (0.01 = 1% net after fees), not percent points.
- `--price` / `limit_price` overrides fee-aware `target_return` sizing.
- Requires TWS/IB Gateway API. Prefer paper port **7497**.
- Last buy comes from recent buy fills when available, else position `avgCost`.
- Restart `tradingai-ibkr` MCP after env changes.
