---
name: sell
description: >-
  Submits an IBKR TWS LIMIT sell (default: full open position, fee-aware limit
  at TRADINGAI_OPEN_DIP_TARGET_RETURN on last buy) via tradingai-ibkr
  sell_stock_limit or tradingai-sell CLI. Use when the user invokes /sell or
  asks to limit-sell a ticker. For TRAIL LIMIT use /sell-trail /
  sell_stock_trail / tradingai-sell-trail.
disable-model-invocation: true
---

# /sell — IBKR TWS limit sell

## Goal

Submit a sell transaction as a **LIMIT** order (not trailing). Arguments: stock
ticker, share count (default all open), and limit price (default
fee-aware `TRADINGAI_OPEN_DIP_TARGET_RETURN` on last buy / avg cost).

## Defaults

| Input | Default |
|-------|---------|
| `order_type` | `LMT` |
| `quantity` | full open long position |
| `limit_price` | fee-aware sell at `target_return` on last buy / avg cost |
| `target_return` | `TRADINGAI_OPEN_DIP_TARGET_RETURN` / ≈**0.3836%** net after fees |
| `dry_run` | **true** unless user explicitly asks to submit live |

## Steps

1. **Resolve inputs** from the user message:
   - `symbol` (required)
   - `quantity` (optional → all open)
   - `limit_price` (optional → fee-aware target)
   - `target_return` (optional → env/default ≈0.3836%; fraction, e.g. `0.01` = 1%)
   - `dry_run` (optional → true)

2. **Prefer MCP** `tradingai-ibkr` → `sell_stock_limit`. Or CLI:

```bash
# Dry-run: full position, fee-aware target limit
tradingai-sell COIN

# Custom net profit % (fraction): 1% after fees
tradingai-sell COIN --target-return 0.01

# Explicit size / limit (skips fee-aware target_return math)
tradingai-sell COIN --qty 33 --price 170.65

# Live (needs TRADINGAI_IBKR_ALLOW_ORDERS=1)
tradingai-sell COIN --submit
tradingai-sell COIN --target-return 0.01 --submit --yes
```

TRAIL LIMIT (separate): `/sell-trail` / `tradingai-sell-trail` / MCP `sell_stock_trail`.

3. **Fail-fast** on `error`, `status == "rejected"`, or missing position when
   qty omitted.

4. **Confirm before live** — print sizing and require confirmation unless the
   user already asked to submit / passed `--yes`.

5. **Present** a short result table (informational — not financial advice):

| Field | Source |
|-------|--------|
| Symbol | `symbol` |
| Side / type | sell / LMT |
| Qty | `quantity` |
| Limit @ | `limit_price` |
| Target return | `sizing.target_return` (when fee-aware) |
| Buy ref | `sizing.buy_price` + source |
| Dry run | `dry_run` |
| Status | `status` / `order_id` |

## Notes

- `target_return` is a **fraction** (0.01 = 1% net after fees), not percent points.
- `--price` / `limit_price` overrides fee-aware `target_return` sizing.
- Requires TWS/IB Gateway API. Prefer paper port **7497**.
- Last buy comes from recent buy fills when available, else position `avgCost`.
- Restart `tradingai-ibkr` MCP after env changes.
