---
name: selltargets
description: >-
  Downloads today's IBKR buy fills via tradingai-ibkr, aggregates them per
  symbol, and computes fee-aware sell prices that realize DEFAULT_TARGET_RETURN
  net profit. Use when the user invokes /selltargets or asks for sell prices /
  profit targets on today's buys.
disable-model-invocation: true
---

# /selltargets — sell prices for today's buys

## Goal

Download today's buy orders from IBKR TWS. For each asset bought today, calculate
the stock price at which selling would realize `DEFAULT_TARGET_RETURN` of net
profit (after IBKR Tiered buy+sell fees). Print those assets with their required
sell prices.

## Constants

- `DEFAULT_TARGET_RETURN` from `tradingai.intraday.profit_target` (also mirrored
  in `tradingai.intraday.config`): `(1/365)*(7/5)` ≈ 0.3836%.
- Use **fee-aware** sell price for that return only — do **not** add
  `DEFAULT_PROFIT_TAKER_BUFFER` unless the user asks.

## Steps

1. **Fetch today's buys** (required first; run from repo root). Prefer the MCP
   tool when available; otherwise call the same backend:

```bash
python - <<'PY'
import json
from tradingai.ibkr.tools.trades import get_last_trades_tool
print(json.dumps(get_last_trades_tool(limit=500, side="buy"), indent=2, default=str))
PY
```

Or MCP: `tradingai-ibkr` → `get_last_trades(limit=500, side="buy")`.

2. **Fail-fast on fetch** — stop if any of these hold:
   - Connection error / empty `error` field with `count == 0` and a connection hint
   - Non-zero process exit and no usable JSON
   - No buy fills at all after filtering to today

3. **Filter to today**: keep fills whose `time` falls on the current calendar day
   in America/New_York (session day). Drop older fills even if IB returns them.

4. **Aggregate per asset** (`symbol`):
   - `shares` = sum of fill shares
   - `buy_notional_usd` = sum of `shares * price` (or `notional_usd`)
   - `buy_avg` = `buy_notional_usd / shares` (VWAP)

5. **Compute sell target** for each asset with the project helper (do not invent
   a raw `buy * (1 + r)` unless the helper is unavailable):

```python
from tradingai.intraday.profit_target import (
    DEFAULT_TARGET_RETURN,
    fee_aware_sell_price,
)

pt = fee_aware_sell_price(
    buy_avg,
    buy_notional_usd=buy_notional_usd,
    net_return=DEFAULT_TARGET_RETURN,
)
sell_at = pt["sell_price"]
```

6. **Sort** assets by `buy_notional_usd` descending (largest buys first).

7. **Present** a concise table (informational only — not financial advice):

| Column | Source |
|--------|--------|
| Symbol | aggregated `symbol` |
| Shares | total shares bought today |
| Buy avg | VWAP fill price |
| Sell @ | `fee_aware_sell_price(...).sell_price` |
| Move % | `price_move_pct` from that helper |
| Target ret % | `DEFAULT_TARGET_RETURN * 100` |
| Est. net $ | `net_profit_usd` from that helper |

Also print one line with the numeric `DEFAULT_TARGET_RETURN` used and the
session day (ET). Note the IBKR account if present. If a symbol has only
fractional leftover sells elsewhere, still use **buy** fills only for this skill.

## Notes

- Requires TWS/IB Gateway with API enabled (`TRADINGAI_IBKR_*` env).
- IB `reqExecutions` is typically session / recent-day only — that matches "today".
- Partial fetch OK only if some buys are returned; connection failure is not OK.
- Do not expand to the full watchlist or include sell fills.
