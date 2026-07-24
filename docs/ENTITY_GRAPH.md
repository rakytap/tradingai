# Entity graph and edge-derived metrics

Mathematical description of how the L3 ticker entity graph and the L2 theme/event graph are used, and how quantifiable metrics are derived from edges.

There are **two graphs**. Quantified edge math for fusion lives mostly on **L2** (theme → sector). The **L3 entity graph** uses typed relations and **prior weights** to rank related tickers; those weights do **not** enter fusion yet.

| Graph | Edge type | Derived quantity | Enters fusion? |
|--------|-----------|------------------|----------------|
| **L3 entity** | Typed, unweighted | Candidate weight \(s(u)\) from relation/commodity/flag priors | No (attached on `raw` only) |
| **L2 theme** | Weighted \(w(\tau,n)\) | Sector exposures \(X(s)\), then `event_score` | Yes |

**Code:** `src/tradingai/news/entity_graph.py`, `src/tradingai/news/analysis/related_candidates.py`, `src/tradingai/news/ontology.py`, `src/tradingai/news/analysis/events.py`.

---

## 1. L3 entity graph

### Graph definition

Treat each ticker as a node \(v \in V\). Outgoing edges are typed, **unweighted** adjacency lists:

\[
E \subseteq V \times V \times R,\quad
R = \{\text{peer},\ \text{supplier},\ \text{customer},\ \text{thematic\_proxy}\}
\]

Node attributes (not edges): commodities \(C(v)\), flags \(\rho(v),\ \phi(v)\) (rate-/FX-sensitive), optional duration bucket.

For subject \(t\), one-hop neighbors are:

\[
N_r(t) = \{ u : (t,u,r) \in E \}
\]

as implemented by `neighbors()` in `entity_graph.py`.

### How L2 themes use L3

Given extracted event themes \(T = \{ \tau_1,\ldots,\tau_m \}\), each theme has an expansion policy \(\Pi(\tau)\) (`THEME_EXPANSIONS`): which relation kinds, commodities, and flags to expand.

Candidate scores use **relation priors** (not learned edge weights):

\[
w_{\text{rel}}(r) =
\begin{cases}
0.8 & r \in \{\text{supplier},\text{customer}\} \\
0.6 & r = \text{peer} \\
0.9 & r = \text{thematic\_proxy}
\end{cases}
\quad
w_{\text{com}}=0.7,\quad
w_{\text{flag}}=0.4
\]

For each theme \(\tau\) and related ticker \(u\), a raw score \(s(u;\tau)\) is assigned by the matching rule (direct edge / commodity cohort / flag). Duplicates keep the **max**:

\[
s(u) = \max_{\tau \in T} s(u;\tau),\qquad
s(u) \leftarrow \mathrm{clip}_{[0,1]}(s(u))
\]

Then candidates are ranked by \((-s(u), u)\) and truncated to `limit` (default 12).

**Default (no themes):** only suppliers + thematic proxies, half-weight:

\[
s(u) = \tfrac12\, w_{\text{rel}}(r)\quad\text{for } r \in \{\text{supplier},\text{thematic\_proxy}\}
\]

**Commodity rule:** if \(u\) shares commodity \(c\) with the theme:

\[
s(u) =
\begin{cases}
\max(w_{\text{com}},\, w_{\text{rel}}(r)) & \text{if also direct edge }(t,u,r) \\
0.85\, w_{\text{com}} & \text{else (cohort only)}
\end{cases}
\]

So the only metrics from edges on L3 today are these **prior weights on relation type** (and the commodity/flag bonuses). They produce:

```text
related_candidates[{ticker, relation, weight, reason, themes}]
```

for debugging / MCP — **not** `AgentSignal.directional_score`.

---

## 2. L2 theme graph (fusion metrics)

This is the weighted graph that **does** drive a scalar news score.

### Graph

Nodes = themes ∪ GICS sectors. Weighted edges \(w(\tau, n)\) from `THEME_EDGES`, scaled by `edge_scale` \(\alpha\):

\[
\tilde w(\tau,n) = \alpha\, w(\tau,n)
\]

Defaults live in `src/tradingai/news/ontology.py`; runtime overrides via `data/event_graph.json` / `tradingai-tune-fusion`.

### Propagation (BFS, multi-source)

Start from event themes with mass \(1\):

\[
(u_0,\ m_0,\ h_0) = (\tau,\ 1,\ 0)
\]

At hop \(h\), along edge \(u \to v\):

\[
m' = m \cdot \tilde w(u,v) \cdot
\begin{cases}
1 & h = 0 \\
\delta & h > 0
\end{cases}
\quad (\delta = \texttt{propagation\_decay},\ \text{default } 0.5)
\]

If \(v\) is a sector node \(s \in S\), accumulate absolute path mass:

\[
W(s) \mathrel{+}= m'
\]

Max hops \(H = 2\) by default. Path identity is `(node, hop)` in the visited set.

So for a single theme \(\tau\), a one-hop sector hit is \(W(s) = \tilde w(\tau,s)\); a two-hop path \(\tau \to u \to s\) contributes \(\tilde w(\tau,u)\,\tilde w(u,s)\,\delta\).

### Sector exposure from events

For each event \(e\) with polarity \(p_e \in \{\pm 1\}\), confidence \(c_e\), themes \(\to\) sector masses \(W_e(s)\):

\[
X(s) = \sum_e p_e\, c_e\, W_e(s)
\]

### Ticker event score

Let \(s^\*\) be the ticker’s sector (canonicalized). With divisor \(D\) (default \(1.5\)):

\[
\text{raw} =
\begin{cases}
X(s^\*) & s^\* \in \mathrm{dom}(X) \\
\frac{1}{|X|}\sum_s X(s) & \text{else if } X \neq \emptyset \\
0 & \text{else}
\end{cases}
\]

\[
\texttt{event\_score} = \mathrm{clip}_{[-1,1]}\!\left(\frac{\text{raw}}{D}\right)
\]

That clipped score is what fusion uses as the news directional signal when events match.

---

## Summary

L3 is a **typed adjacency + prior map** \(R \mapsto w_{\text{rel}}\) used for ranked neighbor expansion. The **path-product / signed exposure** calculus that produces a continuous metric for trading signals is on L2 `THEME_EDGES`, not on the entity-graph edges themselves.
