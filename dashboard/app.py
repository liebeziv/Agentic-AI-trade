"""Atlas Trader — Streamlit dashboard (Phase 1-3)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="Atlas Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- store

@st.cache_resource
def get_store():
    from src.data.data_store import DataStore
    db_path = os.getenv("DUCKDB_PATH", str(BASE_DIR / "data" / "atlas.duckdb"))
    if not Path(db_path).exists():
        return None
    return DataStore(db_path)


# --------------------------------------------------------------- sidebar

def sidebar() -> str:
    with st.sidebar:
        st.title("Atlas Trader")
        st.caption(f"UTC {datetime.utcnow().strftime('%H:%M:%S')}")
        page = st.radio(
            "Navigate",
            ["Overview", "Signals", "Positions", "Risk",
             "Journal", "Attribution", "Agents", "Backtest"],
        )
        st.markdown("---")
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        auto = st.checkbox("Auto-refresh (30s)", value=False)
        if auto:
            import time
            st.caption("Next refresh in 30s…")
            time.sleep(30)
            st.rerun()
    return page


# --------------------------------------------------------------- pages

def page_overview(store) -> None:
    st.title("Portfolio Overview")
    if store is None:
        st.warning("No database found — start `run_paper.py` or `run_multiagent.py` first.")
        return

    end = datetime.utcnow()
    start = end - timedelta(days=30)
    history = store.get_portfolio_history(start, end)

    if history.is_empty():
        st.info("No portfolio snapshots yet. Waiting for first trading cycle…")
        return

    latest = history.row(-1, named=True)
    equity = latest["equity"]
    cash = latest["cash"]
    daily_pnl = latest.get("daily_pnl", 0) or 0
    max_dd = latest.get("max_drawdown", 0) or 0
    initial = history.row(0, named=True)["equity"]
    total_ret = (equity - initial) / max(initial, 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", f"${equity:,.2f}", f"{total_ret:+.1f}%")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Daily P&L", f"${daily_pnl:+.2f}")
    c4.metric("Max Drawdown", f"{max_dd:.1f}%")
    unrealized = equity - cash
    c5.metric("Unrealized P&L", f"${unrealized:+.2f}")

    # Equity curve
    fig = go.Figure()
    ts = history["timestamp"].to_list()
    eq = history["equity"].to_list()
    fig.add_trace(go.Scatter(
        x=ts, y=eq, mode="lines", name="Equity",
        line=dict(color="#2ecc71", width=2),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.08)",
    ))
    fig.update_layout(title="Equity Curve", height=320,
                      xaxis_title="", yaxis_title="USD",
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    # Daily P&L bars
    with col_left:
        if "daily_pnl" in history.columns:
            import pandas as pd
            daily = history.select(["timestamp", "daily_pnl"]).to_pandas()
            daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
            grp = daily.groupby("date")["daily_pnl"].last().reset_index()
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in grp["daily_pnl"]]
            fig2 = go.Figure(go.Bar(
                x=grp["date"], y=grp["daily_pnl"],
                marker_color=colors, name="Daily P&L"
            ))
            fig2.update_layout(title="Daily P&L", height=250, margin=dict(t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    with col_right:
        trades = store.get_trades(start=start, end=end)
        if trades:
            from src.utils.metrics import profit_factor, sharpe_ratio, win_rate
            pnls = [t.pnl for t in trades]
            rets = [t.pnl / max(initial, 1) for t in trades]
            st.subheader("Performance Stats")
            s1, s2 = st.columns(2)
            s1.metric("Total Trades", len(trades))
            s2.metric("Win Rate", f"{win_rate(pnls):.1f}%")
            s1.metric("Profit Factor", f"{profit_factor(pnls):.2f}")
            s2.metric("Sharpe", f"{sharpe_ratio(rets):.2f}")
            avg_win = sum(t.pnl_pct for t in trades if t.pnl > 0) / max(sum(1 for t in trades if t.pnl > 0), 1)
            avg_loss = sum(t.pnl_pct for t in trades if t.pnl <= 0) / max(sum(1 for t in trades if t.pnl <= 0), 1)
            s1.metric("Avg Win", f"{avg_win:.2f}%")
            s2.metric("Avg Loss", f"{avg_loss:.2f}%")


def page_signals(store) -> None:
    st.title("Live Signals")
    if store is None:
        st.warning("No database found.")
        return

    end = datetime.utcnow()
    start = end - timedelta(hours=24)

    rows = store.conn.execute("""
        SELECT instrument, timestamp, technical_score, sentiment_score,
               regime_score, claude_confidence, composite_score, action
        FROM signals WHERE timestamp >= ?
        ORDER BY timestamp DESC LIMIT 300
    """, [start]).fetchall()

    if not rows:
        st.info("No signals yet.")
        return

    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "instrument", "timestamp", "technical_score", "sentiment_score",
        "regime_score", "claude_confidence", "composite_score", "action"
    ])
    latest = df.groupby("instrument").first().reset_index()

    st.subheader("Current Signal Scores")
    cols = st.columns(min(len(latest), 4))
    for i, (_, row) in enumerate(latest.iterrows()):
        score = row["composite_score"]
        col = cols[i % len(cols)]
        delta_color = "normal" if score > 0 else "inverse"
        col.metric(
            label=f"{row['instrument']} — {row['action']}",
            value=f"{score:+.3f}",
            delta=f"Tech:{row['technical_score']:+.2f} Sent:{row['sentiment_score']:+.2f}",
        )

    # Factor breakdown radar
    st.subheader("Factor Breakdown (latest signal per instrument)")
    if len(latest) > 0:
        categories = ["technical_score", "sentiment_score", "regime_score", "claude_confidence"]
        fig = go.Figure()
        for _, row in latest.iterrows():
            vals = [row[c] for c in categories] + [row[categories[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=row["instrument"]
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[-1, 1])),
                          title="Signal Factor Radar", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Time series
    fig2 = px.scatter(df, x="timestamp", y="composite_score",
                      color="instrument", symbol="action",
                      title="Composite Score History (24h)", height=350)
    fig2.add_hline(y=0.25, line_dash="dash", line_color="green",
                   annotation_text="Buy")
    fig2.add_hline(y=-0.25, line_dash="dash", line_color="red",
                   annotation_text="Sell")
    st.plotly_chart(fig2, use_container_width=True)


def page_positions(store) -> None:
    st.title("Open Positions")
    if store is None:
        st.warning("No database found.")
        return

    snap = store.conn.execute("""
        SELECT positions_json, equity FROM portfolio_snapshots
        ORDER BY timestamp DESC LIMIT 1
    """).fetchone()

    if not snap or not snap[0]:
        st.info("No open positions.")
        return

    positions = json.loads(snap[0])
    equity = snap[1] or 1

    if not positions:
        st.info("No open positions.")
        return

    import pandas as pd
    df = pd.DataFrame(positions)
    total_unreal = sum(p.get("unrealized_pnl", 0) for p in positions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Open Positions", len(positions))
    c2.metric("Unrealized P&L", f"${total_unreal:+.2f}",
              f"{total_unreal/equity*100:+.2f}%")
    c3.metric("Equity", f"${equity:,.2f}")

    # Colour unrealized PnL column
    display_cols = [c for c in ["instrument", "side", "quantity",
                                 "entry_price", "current_price",
                                 "unrealized_pnl", "stop_loss", "take_profit"]
                    if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

    if len(positions) > 0 and "unrealized_pnl" in df.columns:
        fig = px.bar(df, x="instrument", y="unrealized_pnl",
                     color="unrealized_pnl",
                     color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],
                     title="Unrealized P&L by Position")
        st.plotly_chart(fig, use_container_width=True)


def page_risk(store) -> None:
    st.title("Risk Monitor")
    if store is None:
        st.warning("No database found.")
        return

    latest = store.conn.execute("""
        SELECT equity, max_drawdown, daily_pnl, positions_json
        FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1
    """).fetchone()

    if not latest:
        st.info("No data yet.")
        return

    equity, max_dd, daily_pnl, pos_json = latest
    positions = json.loads(pos_json or "[]")
    total_risk = sum(
        abs(p.get("entry_price", 0) - (p.get("stop_loss") or p.get("entry_price", 0)))
        * p.get("quantity", 0)
        for p in positions if p.get("stop_loss")
    )
    heat_pct = total_risk / max(equity or 1, 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Drawdown", f"{max_dd or 0:.1f}%",
              delta_color="inverse" if (max_dd or 0) > 10 else "off")
    c2.metric("Daily P&L", f"${daily_pnl or 0:+.2f}",
              delta_color="inverse" if (daily_pnl or 0) < -1000 else "off")
    c3.metric("Portfolio Heat", f"{heat_pct:.1f}%",
              delta_color="inverse" if heat_pct > 15 else "off")
    c4.metric("Open Positions", len(positions))

    # Drawdown history
    end = datetime.utcnow()
    hist = store.get_portfolio_history(end - timedelta(days=30), end)
    if not hist.is_empty():
        dd_vals = [-(v or 0) for v in hist["max_drawdown"].to_list()]
        fig = go.Figure(go.Scatter(
            x=hist["timestamp"].to_list(), y=dd_vals,
            fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
            line=dict(color="#e74c3c"), name="Drawdown"
        ))
        fig.add_hline(y=-10, line_dash="dash", line_color="orange",
                      annotation_text="Soft limit -10%")
        fig.add_hline(y=-15, line_dash="dash", line_color="red",
                      annotation_text="Hard limit -15%")
        fig.update_layout(title="Drawdown (30 days)", height=280,
                          yaxis_title="DD %", margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)


def page_journal(store) -> None:
    st.title("Trade Journal")
    if store is None:
        st.warning("No database found.")
        return

    end = datetime.utcnow()
    start = end - timedelta(days=90)
    trades = store.get_trades(start=start, end=end)

    if not trades:
        st.info("No trades recorded yet.")
        return

    import pandas as pd
    rows = [{
        "Instrument": t.instrument,
        "Side": t.side.value,
        "Entry": f"{t.entry_price:.4g}",
        "Exit": f"{t.exit_price:.4g}",
        "P&L": t.pnl,
        "P&L%": t.pnl_pct,
        "Duration(m)": round(t.duration_minutes),
        "Exit": t.exit_reason.value,
        "Lesson": (t.lessons_learned or "")[:60],
    } for t in trades]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        pnls = [t.pnl for t in trades]
        fig = px.histogram(x=pnls, nbins=30, title="P&L Distribution ($)",
                           color_discrete_sequence=["#3498db"])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        reasons = {}
        for t in trades:
            r = t.exit_reason.value
            reasons[r] = reasons.get(r, 0) + 1
        fig2 = px.pie(values=list(reasons.values()), names=list(reasons.keys()),
                      title="Exit Reasons")
        st.plotly_chart(fig2, use_container_width=True)


def page_attribution(store) -> None:
    st.title("Attribution Analysis")
    if store is None:
        st.warning("No database found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Period (days)", 7, 90, 30)
    with col2:
        if st.button("Run Attribution"):
            st.session_state["run_attribution"] = True

    if not st.session_state.get("run_attribution"):
        st.info("Click 'Run Attribution' to analyse factor contributions.")
        return

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    from src.reflection.attribution import AttributionEngine
    engine = AttributionEngine(store)
    report = engine.generate_report(start, end)

    if "error" in report:
        st.warning(report["error"])
        return

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", report["total_trades"])
    c2.metric("Win Rate", f"{report['win_rate_pct']:.1f}%")
    c3.metric("Total P&L", f"${report['total_pnl']:+.2f}")
    c4.metric("Best Factor", report["best_factor"].title())

    # Factor bar chart
    attrs = report["avg_attribution"]
    factors = list(attrs.keys())
    values = list(attrs.values())
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]
    fig = go.Figure(go.Bar(
        x=factors, y=values, marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
    ))
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(title="Average Factor Attribution",
                      yaxis_title="Signed Contribution", height=320,
                      margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    # Instrument breakdown
    with col_l:
        st.subheader("By Instrument")
        inst = report.get("instrument_breakdown", {})
        if inst:
            inst_df = [{"Instrument": k, "P&L": v["pnl"],
                         "Trades": v["trades"],
                         "WR%": round(v["wins"]/max(v["trades"],1)*100, 1)}
                        for k, v in inst.items()]
            import pandas as pd
            st.dataframe(pd.DataFrame(inst_df).sort_values("P&L", ascending=False),
                         use_container_width=True)

    # Regime breakdown
    with col_r:
        st.subheader("By Regime")
        reg = report.get("regime_breakdown", {})
        if reg:
            reg_df = [{"Regime": k, "P&L": v["pnl"],
                        "Trades": v["trades"],
                        "WR%": round(v["wins"]/max(v["trades"],1)*100, 1)}
                       for k, v in reg.items()]
            import pandas as pd
            st.dataframe(pd.DataFrame(reg_df).sort_values("P&L", ascending=False),
                         use_container_width=True)

    # Lessons
    if report.get("top_lessons"):
        st.subheader("Claude Lessons Learned")
        for lesson in report["top_lessons"]:
            st.info(lesson)


def page_agents(store) -> None:
    st.title("Multi-Agent Status (Phase 3)")

    st.info(
        "Start the multi-agent system with:\n"
        "```\nPYTHONPATH=. python scripts/run_multiagent.py\n```"
    )

    # Show agent configs
    from src.strategy.strategies.momentum_agent import MomentumAgent
    from src.strategy.strategies.mean_reversion_agent import MeanReversionAgent

    agents_info = [
        {
            "Agent": "Momentum",
            "Strategy": "RSI + MACD + MA + ADX filter",
            "Best Regimes": "STRONG_TREND_UP, STRONG_TREND_DOWN, BREAKOUT",
            "Targets": "US Stocks, Futures",
            "Min ADX": 20,
        },
        {
            "Agent": "Mean Reversion",
            "Strategy": "RSI extremes + BB touch + Stochastic",
            "Best Regimes": "RANGE_BOUND, WEAK_TREND",
            "Targets": "Forex, US Stocks",
            "Max ADX": 22,
        },
    ]

    import pandas as pd
    st.subheader("Registered Agents")
    st.dataframe(pd.DataFrame(agents_info), use_container_width=True)

    # Capital allocation simulation
    st.subheader("Capital Allocation Preview")
    st.caption("Simulated fitness-based allocation for different regimes")

    regimes = ["STRONG_TREND_UP", "RANGE_BOUND", "STRONG_TREND_DOWN", "BREAKOUT"]
    alloc_data = {
        "Regime": regimes,
        "Momentum %": [80, 20, 75, 70],
        "Mean Reversion %": [20, 80, 25, 30],
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Momentum", x=regimes, y=alloc_data["Momentum %"],
                         marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Mean Reversion", x=regimes,
                         y=alloc_data["Mean Reversion %"],
                         marker_color="#e67e22"))
    fig.update_layout(barmode="stack", title="Simulated Capital Allocation by Regime",
                      yaxis_title="%", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Live agent signals (from signals table, distinguish by composite score patterns)
    if store:
        end = datetime.utcnow()
        recent_sigs = store.conn.execute("""
            SELECT instrument, timestamp, composite_score, action, regime_score
            FROM signals WHERE timestamp >= ?
            ORDER BY timestamp DESC LIMIT 50
        """, [end - timedelta(hours=4)]).fetchall()

        if recent_sigs:
            st.subheader("Recent Signals (last 4h)")
            import pandas as pd
            sig_df = pd.DataFrame(recent_sigs, columns=[
                "instrument", "timestamp", "composite_score", "action", "regime_score"
            ])
            st.dataframe(sig_df, use_container_width=True)


def page_backtest(store) -> None:
    st.title("Backtest")

    col1, col2, col3 = st.columns(3)
    with col1:
        instrument = st.selectbox("Instrument",
            ["AAPL", "NVDA", "SPY", "GC", "EURUSD", "BTC/USDT", "MSFT", "TSLA"])
    with col2:
        days = st.slider("Lookback (days)", 20, 59, 45)
    with col3:
        walk_forward = st.checkbox("Walk-Forward")

    if st.button("Run Backtest"):
        import asyncio
        import structlog, logging
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL))

        from scripts.run_backtest import load_config, run_backtest, walk_forward_backtest

        config = load_config()
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        with st.spinner(f"Running {instrument} backtest ({days}d)…"):
            if walk_forward:
                windows = asyncio.run(walk_forward_backtest(
                    instrument, config, start, end, train_days=30, test_days=15
                ))
                for s, e, r in windows:
                    st.markdown(f"**Window {s.date()} → {e.date()}**")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Trades", r.total_trades)
                    mc2.metric("Win Rate", f"{r.win_rate:.1f}%")
                    mc3.metric("P&L", f"${r.total_pnl:+.2f}")
                    mc4.metric("Sharpe", f"{r.sharpe_ratio:.2f}")
            else:
                r = asyncio.run(run_backtest(instrument, start, end, config))
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Trades", r.total_trades)
                mc2.metric("Win Rate", f"{r.win_rate:.1f}%")
                mc3.metric("P&L", f"${r.total_pnl:+.2f} ({r.total_pnl_pct:.1f}%)")
                mc4.metric("Sharpe", f"{r.sharpe_ratio:.2f}")
                mc5.metric("Max DD", f"{r.max_drawdown_pct:.1f}%")

                if r.equity_curve:
                    fig = px.line(y=r.equity_curve, title=f"{instrument} Equity Curve",
                                  labels={"x": "Bar", "y": "Equity ($)"})
                    fig.update_traces(line_color="#2ecc71")
                    st.plotly_chart(fig, use_container_width=True)

                if r.trade_log:
                    import pandas as pd
                    tdf = pd.DataFrame([{
                        "Side": t.side.value, "Entry": f"{t.entry_price:.4g}",
                        "Exit": f"{t.exit_price:.4g}",
                        "P&L": f"${t.pnl:+.2f}", "P&L%": f"{t.pnl_pct:+.2f}%",
                        "Exit Reason": t.exit_reason.value,
                    } for t in r.trade_log])
                    st.subheader("Trade Log")
                    st.dataframe(tdf, use_container_width=True)


# --------------------------------------------------------------- main

def main() -> None:
    store = get_store()
    page = sidebar()

    pages = {
        "Overview": page_overview,
        "Signals": page_signals,
        "Positions": page_positions,
        "Risk": page_risk,
        "Journal": page_journal,
        "Attribution": page_attribution,
        "Agents": page_agents,
        "Backtest": page_backtest,
    }
    pages.get(page, page_overview)(store)


if __name__ == "__main__":
    main()
