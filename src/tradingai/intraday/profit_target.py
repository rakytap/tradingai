from __future__ import annotations

from tradingai.predict.ibkr_fees import estimate_order_commission

# Weekly-ish equity premium proxy: (1/365) * (7/5) ≈ 0.3836%
DEFAULT_TARGET_RETURN = (1.0 / 365.0) * (7.0 / 5.0)
DEFAULT_BUY_NOTIONAL_USD = 20_000.0
# Extra cushion on the attached profit-taker beyond target_return (+ fees).
DEFAULT_PROFIT_TAKER_BUFFER = 0.0002  # 0.02%


def fee_aware_sell_price(
    buy_price: float,
    *,
    buy_notional_usd: float,
    net_return: float,
    monthly_volume_shares: float = 0.0,
) -> dict:
    """Sell price that nets ``net_return`` on notional after IBKR Tiered buy+sell fees."""
    price = float(buy_price)
    notional = float(buy_notional_usd)
    ret = float(net_return)
    if price <= 0.0 or notional <= 0.0:
        raise ValueError("buy_price and buy_notional_usd must be positive.")

    shares = notional / price
    buy_commission = estimate_order_commission(
        shares,
        notional,
        monthly_volume_shares=monthly_volume_shares,
    )
    required_net_proceeds = notional * (1.0 + ret) + buy_commission

    sell_price = required_net_proceeds / shares
    sell_commission = 0.0
    for _ in range(8):
        trade_value = shares * sell_price
        sell_commission = estimate_order_commission(
            shares,
            trade_value,
            monthly_volume_shares=monthly_volume_shares + shares,
        )
        sell_price = (required_net_proceeds + sell_commission) / shares

    trade_value = shares * sell_price
    sell_commission = estimate_order_commission(
        shares,
        trade_value,
        monthly_volume_shares=monthly_volume_shares + shares,
    )
    net_proceeds = trade_value - sell_commission
    net_profit = net_proceeds - notional - buy_commission
    return {
        "buy_price": price,
        "buy_notional_usd": notional,
        "shares": shares,
        "net_return": ret,
        "buy_commission_usd": buy_commission,
        "sell_commission_usd": sell_commission,
        "sell_price": sell_price,
        "price_move_pct": (sell_price / price - 1.0) * 100.0,
        "net_profit_usd": net_profit,
    }


def sell_trail_pct(
    *,
    target_return: float = DEFAULT_TARGET_RETURN,
    sell_trail: float | None = None,
) -> dict:
    """Suggested sell trailing-stop distance as fraction and percent.

    Defaults to ``target_return / 2`` when ``sell_trail`` is omitted.
    """
    ret = float(target_return)
    trail = float(sell_trail) if sell_trail is not None else ret / 2.0
    if trail < 0.0:
        raise ValueError("sell_trail must be non-negative.")
    return {
        "sell_trail": trail,
        "sell_trail_pct": round(trail * 100.0, 6),
    }


def trail_stop_price(ref_price: float, *, sell_trail: float) -> float:
    """Price a ``sell_trail`` fraction below ``ref_price``."""
    price = float(ref_price)
    trail = float(sell_trail)
    if price <= 0.0:
        raise ValueError("ref_price must be positive.")
    if trail < 0.0:
        raise ValueError("sell_trail must be non-negative.")
    return price * (1.0 - trail)


def target_sell_price(
    buy_price: float,
    *,
    buy_notional_usd: float = DEFAULT_BUY_NOTIONAL_USD,
    target_return: float = DEFAULT_TARGET_RETURN,
    profit_taker_buffer: float = DEFAULT_PROFIT_TAKER_BUFFER,
    sell_trail: float | None = None,
    monthly_volume_shares: float = 0.0,
) -> dict:
    """Attached profit-taker + trailing-sell levels for an open-dip entry.

    Profit taker: fee-aware sell price for
    ``target_return + profit_taker_buffer`` (default buffer 0.02%).

    Trail: activates at the fee-aware sell price for ``target_return / 2``;
    trailing distance defaults to ``target_return / 2``.
    """
    price = float(buy_price)
    notional = float(buy_notional_usd)
    ret = float(target_return)
    buffer = max(0.0, float(profit_taker_buffer))
    if price <= 0.0 or notional <= 0.0:
        raise ValueError("buy_price and buy_notional_usd must be positive.")

    # Profit taker: target return + fees (via solver) + buffer.
    taker_return = ret + buffer
    taker = fee_aware_sell_price(
        price,
        buy_notional_usd=notional,
        net_return=taker_return,
        monthly_volume_shares=monthly_volume_shares,
    )

    # Trail activates at half target return (fees included in price).
    activate_return = ret / 2.0
    activate = fee_aware_sell_price(
        price,
        buy_notional_usd=notional,
        net_return=activate_return,
        monthly_volume_shares=monthly_volume_shares,
    )

    trail = sell_trail_pct(target_return=ret, sell_trail=sell_trail)
    trail_stop_at_activation = trail_stop_price(
        activate["sell_price"],
        sell_trail=trail["sell_trail"],
    )

    return {
        "buy_price": round(price, 6),
        "buy_notional_usd": round(notional, 2),
        "shares": round(taker["shares"], 6),
        "target_return": ret,
        "target_return_pct": round(ret * 100.0, 6),
        "buy_commission_usd": round(taker["buy_commission_usd"], 4),
        "sell_commission_usd": round(taker["sell_commission_usd"], 4),
        "total_commission_usd": round(
            taker["buy_commission_usd"] + taker["sell_commission_usd"], 4
        ),
        # Profit-taker attached order
        "profit_taker_buffer": buffer,
        "profit_taker_buffer_pct": round(buffer * 100.0, 6),
        "profit_taker_return": taker_return,
        "profit_taker_return_pct": round(taker_return * 100.0, 6),
        "profit_taker_price": round(taker["sell_price"], 6),
        "profit_taker_price_move_pct": round(taker["price_move_pct"], 4),
        "profit_taker_net_profit_usd": round(taker["net_profit_usd"], 4),
        # Back-compat alias: target_sell_price == profit taker
        "target_sell_price": round(taker["sell_price"], 6),
        "price_move_pct": round(taker["price_move_pct"], 4),
        "net_profit_usd": round(taker["net_profit_usd"], 4),
        # Trailing sell order
        "trail_activate_return": activate_return,
        "trail_activate_return_pct": round(activate_return * 100.0, 6),
        "trail_activate_price": round(activate["sell_price"], 6),
        "trail_activate_price_move_pct": round(activate["price_move_pct"], 4),
        "sell_trail": trail["sell_trail"],
        "sell_trail_pct": trail["sell_trail_pct"],
        "trail_stop_at_activation_price": round(trail_stop_at_activation, 6),
    }
