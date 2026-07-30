# How `combined_score` is built

End-to-end path from prices and news to the fused score used by predictions and (by default) the paper portfolio.

```mermaid
flowchart TD
  prices[OHLCV time series] --> feats[Technical + RS features]
  feats --> l5["L5: sector CS ML who wins inside a sector"]
  l5 --> techDir["technical directional_score"]

  headlines[News / filings / social] --> events["L2: typed events themes sector exposure"]
  events --> eventScore[event_score]
  events --> l3cands["L3: ticker entity graph related candidates + news spill"]
  l3cands --> spill[L3 spill blend]
  eventScore --> spill
  spill --> newsDir["news directional_score"]

  regimeBoard["L4: sector board bullish / neutral / bearish"] --> regDir["regime directional_score"]

  techDir --> fuse[Weighted sum]
  newsDir --> fuse
  regDir --> fuse
  fuse --> combined["combined_score"]
```

| Layer | Meaning in this graph |
|-------|------------------------|
| **L2** | News as typed events → themes → sector exposure → `event_score` |
| **L3** | Ticker entity graph: related names; inbound spill into news score |
| **L4** | Sector board stance that feeds the regime agent |
| **L5** | Cross-sectional ML on prices: peer-beat prob → technical score |

*(L1 is market-level factors behind the regime board; it isn’t a separate node in this fusion path.)*

## Fusion formula

```text
combined_score =
    w_tech   * technical_score
  + w_news   * news_score
  + w_regime * regime_score
```

Weights come from tuned `data/fusion_weights.json` or `TRADINGAI_PREDICT_WEIGHTS` (default ~0.45 / 0.35 / 0.20).

L3 spill blend weight α comes from tuned `l3_news_spill_alpha` in `data/news_scoring.json` (via `tradingai-tune-fusion`) when news scoring load is enabled; otherwise `TRADINGAI_L3_NEWS_SPILL_ALPHA` (default `0.25`).

Assumes `TRADINGAI_PREDICT_BACKEND=ml` for the technical leg (L5). With `fusion`, technical is rule-based indicators instead of the CS model.
