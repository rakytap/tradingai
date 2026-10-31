---
name: buy
description: >-
  Places an IBKR TWS LIMIT buy sized to a cash budget (integer shares, default
  mid price, default spend 100% of total budget, no stop-loss by default)
  via tradingai-ibkr buy_stock_budget. Use when the user invokes /buy or asks
  to buy a ticker on IBKR with a budget.
disable-model-invocation: true
---

# /buy — IBKR TWS budget limit buy

## Goal

Make a stock buy transaction on IBKR TWS. By default the buy transaction should
be a LIMIT transaction. The input should be the name of the ticker and the asset
price. (By default the asset price should be the current mid). Also the input
should be the total budget. The transaction should buy an integer amount of
stocks fitting into the full budget.

## Defaults

| Input | Default |
|-------|---------|
| `order_type` | `LMT` (always for this skill) |
| `limit_price` | current IB mid (`buy_stock_budget` omits price → TWS mid) |
| `total_budget` | `TRADINGAI_OPEN_DIP_BUY_NOTIONAL` / 20000 if user omits |
| `budget_fraction` | **1.0** (spend = 100% of total budget) |
| `stop_loss` | **none** (no SELL STP child unless user asks) |
| `dry_run` | **true** unless user explicitly asks to submit live |

Shares: `floor((total_budget * budget_fraction) / limit_price)`.

### Advanced: stop-loss

By default **no** stop-loss is attached. Opt in with a fraction
(e.g. `stop_loss=0.01` / `--stop-loss 0.01` for 1% under the buy limit:
`stop_price = limit_price * (1 - stop_loss)`).

## Steps

1. **Resolve inputs** from the user message:
   - `symbol` (required)
   - `limit_price` (optional → mid)
   - `total_budget` (optional → config default)
   - `budget_fraction` (optional → 1.0)
   - `stop_loss` (optional → none; only set if user asks)
   - `dry_run` (optional → true)

2. **Prefer MCP** `tradingai-ibkr` → `buy_stock_budget`. Or use the CLI:

```bash
# Dry-run (default): LIMIT @ price, spend full budget, no stop
tradingai-buy COIN --price 170 --budget 6000

# Opt-in stop (1% or 2%)
tradingai-buy COIN --price 170 --budget 6000 --stop-loss 0.01
tradingai-buy COIN --price 170 --budget 6000 --stop-loss 0.02

# Limit at current mid, default budget from env
tradingai-buy AAPL

# Live submit (needs TRADINGAI_IBKR_ALLOW_ORDERS=1; prompts unless --yes)
tradingai-buy COIN --price 170 --budget 6000 --submit
tradingai-buy COIN --price 170 --budget 6000 --submit --yes
```

Fallback Python backend:

```bash
python - <<'PY'
import json
from tradingai.ibkr.tools.orders import buy_stock_budget_tool
print(json.dumps(
    buy_stock_budget_tool(
        symbol="AAPL",          # replace
        total_budget=20000,     # replace / omit for config default
        limit_price=None,       # None → current mid
        budget_fraction=1.0,
        stop_loss=None,         # default: no STP; pass 0.01 for 1%
        dry_run=True,
    ),
    indent=2,
    default=str,
))
PY
```

3. **Fail-fast** if the result has `error`, `status == "rejected"`, or
   `quantity < 1`. Show the message / hint (TWS down, no quote, budget too
   small, `allow_orders` gate).

4. **Confirm before live** — if the user wants `dry_run=false`:
   - Require `TRADINGAI_IBKR_ALLOW_ORDERS=1`
   - Print sizing (`quantity`, `limit_price`, `stop_price` if any,
     `spend_budget`, `notional`) and ask for explicit confirmation unless
     they already said to submit

5. **Present** a short result table (informational — not financial advice):

| Field | Source |
|-------|--------|
| Symbol | `symbol` |
| Side / type | buy / LMT (+ STP child when stop enabled) |
| Limit @ | `limit_price` (note mid source if quoted) |
| Stop @ | `stop_price` / `sizing.stop_price` (omit if no stop) |
| Shares | `quantity` / `sizing.quantity` |
| Spend budget | `sizing.spend_budget` (100% of total by default) |
| Notional | `sizing.notional` |
| Dry run | `dry_run` |
| Status | `status` / `submitted` / `order_id` / `stop_order_id` |

## Notes

- Requires TWS/IB Gateway API (`TRADINGAI_IBKR_*`). Mid needs market data.
- Prefer paper port **7497** while testing.
- Do **not** use `buy_stock` (MKT-oriented) for this skill — use
  `buy_stock_budget`.
- Restart `tradingai-ibkr` MCP after changing order-related env vars.
