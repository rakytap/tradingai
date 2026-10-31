---
name: sell-trail
description: >-
  Submits an IBKR TWS TRAIL LIMIT sell (default: full open position, fee-aware
  limit at TRADINGAI_OPEN_DIP_TARGET_RETURN, 0.05% trail) via tradingai-ibkr
  sell_stock_trail or tradingai-sell-trail CLI. Use when the user invokes
  /sell-trail or asks for a one-shot trailing-limit sell. Does not start a
  background watcher or modify an existing trail.
disable-model-invocation: true
---

# /sell-trail — IBKR TWS trail-limit sell

## Goal

Submit a sell as a one-shot **TRAIL LIMIT** order (not plain LIMIT, not a
stepped watcher). Arguments: ticker, share count (default all open), trail
stop/limit reference (default fee-aware `TRADINGAI_OPEN_DIP_TARGET_RETURN` on
last buy / avg cost), and trail percent (default **0.05%**).

## Defaults

| Input | Default |
|-------|---------|
| `order_type` | `TRAIL LIMIT` |
| `quantity` | full open long position |
| `limit_price` | fee-aware sell at `target_return` on last buy / avg cost |
| `target_return` | `TRADINGAI_OPEN_DIP_TARGET_RETURN` / ≈**0.3836%** net after fees |
| `trail_percent` | **0.05** (IB percent points = 0.05%) |
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
# Dry-run: full position, fee-aware target, 0.05% trail
tradingai-sell-trail COIN

# Custom trail / net profit %
tradingai-sell-trail COIN --trail-percent 0.1 --target-return 0.01

# Explicit size / stop reference
tradingai-sell-trail COIN --qty 33 --price 170.65

# Live (needs TRADINGAI_IBKR_ALLOW_ORDERS=1)
tradingai-sell-trail COIN --submit
tradingai-sell-trail COIN --submit --yes
```

Plain LIMIT: `/sell` / `tradingai-sell` / MCP `sell_stock_limit`.
Stepped profit watcher (separate): `tradingai-sell-trail-watch` / MCP
`sell_trail_start`.

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
| Stop / limit ref | `limit_price` |
| Trail % | `trail_percent` |
| Target return | `sizing.target_return` (when fee-aware) |
| Buy ref | `sizing.buy_price` + source |
| Dry run | `dry_run` |
| Status | `status` / `order_id` |

## Notes

- One order only — do **not** start `sell_trail_start` or tighten trail later.
- `target_return` is a **fraction** (0.01 = 1% net after fees), not percent points.
- `--price` / `limit_price` overrides fee-aware `target_return` sizing.
- Requires TWS/IB Gateway API. Prefer paper port **7497**.
- Restart `tradingai-ibkr` MCP after env changes.
