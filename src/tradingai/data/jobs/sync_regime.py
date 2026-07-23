from __future__ import annotations

from tradingai.data.connection import initialize_schema
from tradingai.data.repositories.regime_snapshots import save_regime_snapshot
from tradingai.regime.analysis.state_builder import build_regime_state


def sync_regime_state() -> dict:
    initialize_schema()
    state = build_regime_state()
    payload = state.to_public_dict()
    snapshot_id = save_regime_snapshot(payload, as_of=state.as_of)
    return {
        "snapshot_id": snapshot_id,
        "as_of": payload["as_of"],
        "market_regime": state.market_regime,
        "market": payload["market"],
        "sectors": payload["sectors"],
        "confidence": payload["confidence"],
    }


def main() -> None:
    result = sync_regime_state()
    print(f"Saved regime snapshot id={result['snapshot_id']} as_of={result['as_of']}")
    print(f"market_regime={result['market_regime']} confidence={result['confidence']}")
    market = result["market"]
    print(
        "factors: "
        f"risk={market['risk']} vol={market['vol']} rates={market['rates']} "
        f"credit={market['credit']} growth_value={market['growth_value']} "
        f"equity_trend={market['equity_trend']}"
    )
    print("sector board:")
    for item in result["sectors"]:
        print(f"  {item['sector']}: {item['stance']} ({item['regime']} via {item['proxy']})")


if __name__ == "__main__":
    main()
