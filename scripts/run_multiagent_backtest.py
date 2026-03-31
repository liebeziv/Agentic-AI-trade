"""Multi-agent backtest — runs all 4 specialist agents with orchestrator over historical data."""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from dotenv import load_dotenv

from scripts.run_backtest import RuleBasedClaudeMock, compute_results, BacktestResults
from src.analysis.regime_detector import RegimeDetector
from src.analysis.technical import TechnicalEngine
from src.data.data_store import DataStore
from src.data.market_data import MarketDataFetcher
from src.execution.paper_engine import PaperTradingEngine
from src.orchestrator.correlation_monitor import CorrelationMonitor
from src.orchestrator.global_risk_manager import GlobalRiskManager
from src.strategy.orchestrator import Orchestrator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.strategy.strategies.event_driven_agent import EventDrivenAgent
from src.strategy.strategies.mean_reversion_agent import MeanReversionAgent
from src.strategy.strategies.momentum_agent import MomentumAgent
from src.strategy.strategies.stat_arb_agent import StatArbAgent
from src.types import (
    Bar, MarketRegime, Order, OrderType, PortfolioState, Side, TradeAction, TradeRecord,
)
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import max_drawdown, sharpe_ratio

load_dotenv()
log = get_logger(__name__)

# Phase 3 exit criteria
PHASE3_SHARPE_TARGET = 2.0
PHASE3_CORRELATION_TARGET = 0.3


@dataclass
class AgentBacktestResult:
    agent_id: str
    name: str
    signals_generated: int = 0
    signals_executed: int = 0
    signals_blocked_risk: int = 0
    signals_blocked_global: int = 0
    signals_dropped_conflict: int = 0
    daily_returns: list[float] = field(default_factory=list)


@dataclass
class MultiAgentBacktestResults:
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)
    instruments: list[str] = field(default_factory=list)
    initial_capital: float = 100_000.0
    portfolio: BacktestResults = field(default_factory=BacktestResults)
    per_agent: dict[str, AgentBacktestResult] = field(default_factory=dict)
    avg_inter_agent_correlation: float = 0.0
    diversification_score: float = 0.5
    phase3_sharpe_met: bool = False
    phase3_correlation_met: bool = False
    phase3_exit_met: bool = False


def load_config() -> dict:
    base = Path(__file__).parent.parent
    with open(base / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(base / "config" / "instruments.yaml") as f:
        instruments_cfg = yaml.safe_load(f)
    with open(base / "config" / "risk_params.yaml") as f:
        risk_cfg = yaml.safe_load(f)
    return {
        "trading": settings["trading"],
        "instruments": instruments_cfg["instruments"],
        "hard_limits": risk_cfg["hard_limits"],
        "soft_limits": risk_cfg["soft_limits"],
        "position_sizing": risk_cfg["position_sizing"],
        "paper": {
            "initial_capital": settings["backtest"]["initial_capital"],
            "commission_pct": settings["backtest"]["commission_pct"],
            "slippage_mean_pct": settings["backtest"]["slippage_pct"],
            "slippage_std_pct": settings["backtest"]["slippage_pct"] / 2,
            "latency_ms": 0,
        },
    }


async def run_multiagent_backtest(
    instruments: list[str],
    start: datetime,
    end: datetime,
    config: dict,
) -> MultiAgentBacktestResults:
    """
    Replay historical bars through all 4 agents + orchestrator.

    For each bar (15m):
    1. Compute technicals for every instrument
    2. Each agent generates signals independently
    3. Orchestrator resolves conflicts
    4. GlobalRiskManager portfolio-level check
    5. Per-agent risk check
    6. Execute via PaperTradingEngine
    """
    store = DataStore(":memory:")
    data_fetcher = MarketDataFetcher(config["trading"], store)
    tech_engine = TechnicalEngine()
    claude_mock = RuleBasedClaudeMock()
    regime_detector = RegimeDetector()
    risk_mgr = RiskManager(config["hard_limits"], config["soft_limits"], store)
    position_sizer = PositionSizer(config["position_sizing"], config["hard_limits"], store)
    paper_engine = PaperTradingEngine(config["paper"], store)
    initial_capital = config["paper"]["initial_capital"]

    agent_cfg = {"instruments": config["instruments"]}
    kwargs = dict(config=agent_cfg, store=store, claude_agent=claude_mock, technical=tech_engine)

    momentum    = MomentumAgent(agent_id="momentum", **kwargs)
    mean_rev    = MeanReversionAgent(agent_id="mean_reversion", **kwargs)
    event_drv   = EventDrivenAgent(agent_id="event_driven", **kwargs)
    stat_arb    = StatArbAgent(agent_id="stat_arb", **kwargs)
    agents      = [momentum, mean_rev, event_drv, stat_arb]

    orchestrator = Orchestrator(agents, initial_capital)
    global_risk  = GlobalRiskManager(config["hard_limits"])
    corr_monitor = CorrelationMonitor(agents)

    agent_stats: dict[str, AgentBacktestResult] = {
        a.agent_id: AgentBacktestResult(a.agent_id, a.name) for a in agents
    }

    # yfinance: 15m data available for last 60 days only
    now = datetime.utcnow()
    start_15m = max(start, now - timedelta(days=59))

    # Fetch historical data for all instruments
    log.info("Fetching data", instruments=instruments, start=start, end=end)
    bars_15m_map: dict[str, list[dict]] = {}
    bars_1h_map: dict[str, pl.DataFrame] = {}

    for instrument in instruments:
        try:
            df_15m = await data_fetcher.get_historical(instrument, "15m", start_15m, end)
            df_1h  = await data_fetcher.get_historical(instrument, "1h", start, end)
            if not df_15m.is_empty():
                bars_15m_map[instrument] = df_15m.to_dicts()
            if not df_1h.is_empty():
                bars_1h_map[instrument] = df_1h
            log.info("Data fetched", instrument=instrument,
                     bars_15m=len(bars_15m_map.get(instrument, [])),
                     bars_1h=len(df_1h))
        except Exception as exc:
            log.error("Data fetch failed", instrument=instrument, error=str(exc))

    if not bars_15m_map:
        log.error("No data fetched for any instrument")
        return MultiAgentBacktestResults(
            start_date=start, end_date=end, instruments=instruments,
            initial_capital=initial_capital,
        )

    # Align bar counts — use shortest available instrument
    max_bars = max(len(v) for v in bars_15m_map.values())
    warmup = 200

    all_trades: list[TradeRecord] = []
    equity_curve: list[float] = [initial_capital]
    prev_equity = initial_capital

    log.info("Starting replay", max_bars=max_bars, warmup=warmup, instruments=list(bars_15m_map))

    for i in range(warmup, max_bars):
        # Build technical snapshots for all instruments at bar i
        technical_snapshots: dict = {}
        market_data: dict = {}
        regime: MarketRegime = MarketRegime.RANGE_BOUND

        for instrument, bars_list in bars_15m_map.items():
            if i >= len(bars_list):
                continue
            window = pl.DataFrame(bars_list[:i + 1])
            snap = tech_engine.compute_all(window, instrument, "15m")
            technical_snapshots[instrument] = snap
            market_data[instrument] = window

            # Regime from 1h
            if instrument in bars_1h_map:
                cutoff = bars_list[i]["timestamp"]
                h1_win = bars_1h_map[instrument].filter(
                    pl.col("timestamp") <= cutoff
                ).tail(200)
                if not h1_win.is_empty():
                    regime = regime_detector.classify(h1_win)

        if not technical_snapshots:
            continue

        # Capital allocation (fitness-weighted)
        allocation = await orchestrator.allocate_capital(regime)

        # Each agent generates signals independently
        all_raw_signals = []
        for agent in agents:
            if not agent.is_active or allocation.get(agent.agent_id, 0) <= 0:
                continue
            targets = set(agent.target_instruments)
            tech_filtered = {k: v for k, v in technical_snapshots.items()
                             if not targets or k in targets}
            data_filtered = {k: v for k, v in market_data.items()
                             if not targets or k in targets}
            try:
                sigs = await agent.generate_signals(data_filtered, tech_filtered, [], regime)
                agent_stats[agent.agent_id].signals_generated += len(sigs)
                # Tag each signal with source agent for tracking
                for sig in sigs:
                    sig._agent_id = agent.agent_id   # lightweight tag
                all_raw_signals.extend(sigs)
            except Exception as exc:
                log.debug("Agent signal error", agent=agent.agent_id, error=str(exc))

        # Conflict resolution
        resolved = orchestrator.resolve_conflicts(all_raw_signals)

        # Track dropped signals
        raw_instruments = {s.instrument for s in all_raw_signals}
        resolved_instruments = {s.instrument for s in resolved}
        dropped_instruments = raw_instruments - resolved_instruments
        for sig in all_raw_signals:
            if sig.instrument in dropped_instruments:
                agent_id = getattr(sig, "_agent_id", "unknown")
                if agent_id in agent_stats:
                    agent_stats[agent_id].signals_dropped_conflict += 1

        # Portfolio state
        portfolio = paper_engine.portfolio

        # Execute resolved signals
        for signal in resolved:
            if signal.action in (TradeAction.NEUTRAL, TradeAction.NO_TRADE):
                continue

            agent_id = getattr(signal, "_agent_id", "unknown")

            # Per-strategy risk check
            risk_result = risk_mgr.check_signal(signal, portfolio, [])
            if not risk_result.approved:
                if agent_id in agent_stats:
                    agent_stats[agent_id].signals_blocked_risk += 1
                continue

            # Portfolio-level risk check
            global_result = global_risk.check_portfolio_signal(
                signal, agent_id, [], portfolio
            )
            if not global_result.approved:
                if agent_id in agent_stats:
                    agent_stats[agent_id].signals_blocked_global += 1
                continue

            # Size
            snap = technical_snapshots.get(signal.instrument)
            atr = snap.indicators.get("atr_14", 0.0) if snap else 0.0
            current_close = snap.indicators.get("close", 0.0) if snap else 0.0
            size = position_sizer.calculate(
                signal, portfolio, {"current_price": current_close}, atr
            )
            if size.units <= 0:
                continue

            # Build and execute order
            order_side = (Side.BUY if signal.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
                          else Side.SELL)
            instrument = signal.instrument
            bars_list = bars_15m_map.get(instrument, [])
            if i >= len(bars_list):
                continue
            row = bars_list[i]
            current_bar = Bar(
                timestamp=row["timestamp"],
                open=row["open"], high=row["high"],
                low=row["low"], close=row["close"],
                volume=row["volume"], instrument=instrument, timeframe="15m",
            )

            # Get recommendation for SL/TP from Claude mock
            rec = await claude_mock.analyze_market(
                instrument=instrument,
                technical={"15m": snap} if snap else {},
                news=[], regime=regime,
                recent_trades=[], portfolio_state=portfolio,
            )

            order = Order(
                id=f"MB-{uuid.uuid4().hex[:8]}",
                instrument=instrument,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=size.units,
                stop_loss=rec.stop_loss if rec else None,
                take_profit=rec.take_profit_1 if rec else None,
            )

            try:
                await paper_engine.execute_order(order, current_bar)
                risk_mgr.record_trade_executed(instrument)
                if agent_id in agent_stats:
                    agent_stats[agent_id].signals_executed += 1
            except Exception as exc:
                log.debug("Order execution error", error=str(exc))

        # Check stops for all open positions
        for instrument, bars_list in bars_15m_map.items():
            if i >= len(bars_list):
                continue
            row = bars_list[i]
            bar = Bar(
                timestamp=row["timestamp"],
                open=row["open"], high=row["high"],
                low=row["low"], close=row["close"],
                volume=row["volume"], instrument=instrument, timeframe="15m",
            )
            closed = paper_engine.check_stops(bar)
            all_trades.extend(closed)

        # Update mark-to-market
        current_prices = {}
        for instrument, bars_list in bars_15m_map.items():
            if i < len(bars_list):
                current_prices[instrument] = bars_list[i]["close"]
        paper_engine.update_mark_to_market(current_prices)

        # Equity curve
        equity_curve.append(paper_engine.portfolio.equity)

        # Daily returns for correlation tracking (every 26 bars ≈ 1 trading day at 15m)
        if i % 26 == 0:
            daily_ret = (paper_engine.portfolio.equity - prev_equity) / max(prev_equity, 1)
            prev_equity = paper_engine.portfolio.equity
            for agent in agents:
                agent.performance.daily_returns.append(daily_ret)

    # Force-close remaining positions at last known prices
    last_prices = {}
    for instrument, bars_list in bars_15m_map.items():
        if bars_list:
            last_prices[instrument] = bars_list[-1]["close"]
    if last_prices:
        closed = paper_engine.force_close_all(last_prices)
        all_trades.extend(closed)

    # Compute portfolio-level results
    portfolio_results = compute_results(all_trades, equity_curve, initial_capital)

    # Compute inter-agent correlation
    corr_matrix = corr_monitor.compute_correlation_matrix(lookback_days=30)
    div_score = corr_monitor.get_diversification_score(lookback_days=30)

    # Average off-diagonal correlation
    avg_corr = 0.0
    if corr_matrix:
        ids = list(corr_matrix.keys())
        n = len(ids)
        pairs = [(ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)]
        if pairs:
            avg_corr = float(np.mean([abs(corr_matrix[a][b]) for a, b in pairs]))

    phase3_sharpe_met = portfolio_results.sharpe_ratio >= PHASE3_SHARPE_TARGET
    phase3_corr_met   = avg_corr <= PHASE3_CORRELATION_TARGET

    return MultiAgentBacktestResults(
        start_date=start,
        end_date=end,
        instruments=instruments,
        initial_capital=initial_capital,
        portfolio=portfolio_results,
        per_agent=agent_stats,
        avg_inter_agent_correlation=round(avg_corr, 4),
        diversification_score=round(div_score, 4),
        phase3_sharpe_met=phase3_sharpe_met,
        phase3_correlation_met=phase3_corr_met,
        phase3_exit_met=phase3_sharpe_met and phase3_corr_met,
    )


def print_results(r: MultiAgentBacktestResults) -> None:
    print(f"\n{'='*65}")
    print("  MULTI-AGENT BACKTEST - PORTFOLIO RESULTS")
    print(f"  {r.start_date.date()} to {r.end_date.date()}")
    print(f"  Instruments: {', '.join(r.instruments)}")
    print(f"{'='*65}")
    p = r.portfolio
    print(f"  Total Trades:        {p.total_trades:>8d}")
    print(f"  Win Rate:            {p.win_rate:>7.1f}%")
    print(f"  Total P&L:           ${p.total_pnl:>10.2f}  ({p.total_pnl_pct:.1f}%)")
    print(f"  Profit Factor:       {p.profit_factor:>9.2f}")
    print(f"  Sharpe Ratio:        {p.sharpe_ratio:>9.2f}")
    print(f"  Sortino Ratio:       {p.sortino_ratio:>9.2f}")
    print(f"  Max Drawdown:        {p.max_drawdown_pct:>7.1f}%")
    print(f"  Calmar Ratio:        {p.calmar_ratio:>9.2f}")

    print(f"\n{'─'*65}")
    print("  PER-AGENT SIGNAL ACTIVITY")
    print(f"  {'Agent':<18} {'Generated':>10} {'Executed':>10} {'Blk Risk':>9} {'Blk GRM':>9} {'Conflict':>9}")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*9} {'─'*9} {'─'*9}")
    for aid, stats in r.per_agent.items():
        print(f"  {stats.name:<18} {stats.signals_generated:>10} {stats.signals_executed:>10} "
              f"{stats.signals_blocked_risk:>9} {stats.signals_blocked_global:>9} "
              f"{stats.signals_dropped_conflict:>9}")

    print(f"\n{'─'*65}")
    print("  CORRELATION & DIVERSIFICATION")
    print(f"  Avg Inter-Agent Correlation: {r.avg_inter_agent_correlation:.4f}")
    print(f"  Diversification Score:       {r.diversification_score:.4f}")

    print(f"\n{'─'*65}")
    print("  PHASE 3 EXIT CRITERIA")
    sharpe_mark = "OK" if r.phase3_sharpe_met else "--"
    corr_mark   = "OK" if r.phase3_correlation_met else "--"
    exit_mark   = "PASSED" if r.phase3_exit_met else "NOT MET"
    print(f"  [{sharpe_mark}] Portfolio Sharpe > {PHASE3_SHARPE_TARGET:.1f}  "
          f"(actual: {r.portfolio.sharpe_ratio:.2f})")
    print(f"  [{corr_mark}] Avg Correlation < {PHASE3_CORRELATION_TARGET:.1f}  "
          f"(actual: {r.avg_inter_agent_correlation:.3f})")
    print(f"\n  Result: {exit_mark}")
    print(f"{'='*65}\n")


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Backtest")
    parser.add_argument("--instruments", nargs="+",
                        default=["AAPL", "MSFT", "NVDA", "SPY"],
                        help="Instruments to backtest")
    parser.add_argument("--days", type=int, default=59,
                        help="Lookback days (max 59 for 15m data via yfinance)")
    parser.add_argument("--log-level", default="WARNING",
                        help="Log level (WARNING suppresses per-bar noise)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()

    end   = datetime.utcnow()
    start = end - timedelta(days=args.days)

    results = await run_multiagent_backtest(args.instruments, start, end, config)
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
