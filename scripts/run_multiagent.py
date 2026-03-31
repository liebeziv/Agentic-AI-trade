"""Multi-agent trading loop (Phase 3) — orchestrates specialist agents."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.analysis.claude_agent import ClaudeAgent
from src.analysis.regime_detector import RegimeDetector
from src.analysis.technical import TechnicalEngine
from src.data.data_store import DataStore
from src.data.market_data import MarketDataFetcher
from src.execution.exchange_adapter import AdapterFactory
from src.execution.order_manager import KillSwitch, OrderManager
from src.reflection.attribution import AttributionEngine
from src.reflection.trade_journal import TradeJournal
from src.strategy.orchestrator import Orchestrator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.strategy.strategies.mean_reversion_agent import MeanReversionAgent
from src.strategy.strategies.momentum_agent import MomentumAgent
from src.strategy.strategies.event_driven_agent import EventDrivenAgent
from src.strategy.strategies.stat_arb_agent import StatArbAgent
from src.orchestrator.global_risk_manager import GlobalRiskManager
from src.orchestrator.correlation_monitor import CorrelationMonitor
from src.reflection.strategy_evolver import StrategyEvolver
from src.types import Order, OrderStatus, OrderType, Side, TradeAction
from src.utils.logger import get_logger, setup_logging
from src.utils.notifier import Notifier
from src.utils.time_utils import is_market_open

load_dotenv()
log = get_logger(__name__)


def load_config() -> dict:
    base = Path(__file__).parent.parent
    with open(base / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(base / "config" / "instruments.yaml") as f:
        instruments_cfg = yaml.safe_load(f)
    with open(base / "config" / "risk_params.yaml") as f:
        risk_cfg = yaml.safe_load(f)
    return {
        "system": settings["system"],
        "trading": settings["trading"],
        "claude": settings["claude"],
        "instruments": instruments_cfg["instruments"],
        "hard_limits": risk_cfg["hard_limits"],
        "soft_limits": risk_cfg["soft_limits"],
        "position_sizing": risk_cfg["position_sizing"],
        "db_path": os.getenv("DUCKDB_PATH", "./data/atlas.duckdb"),
        "alpaca": {
            "api_key": os.getenv("ALPACA_API_KEY", ""),
            "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
            "paper": True,
        },
        "binance": {"api_key": os.getenv("BINANCE_API_KEY", ""),
                    "secret": os.getenv("BINANCE_SECRET", ""), "testnet": True},
        "notifier": {"telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
                     "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "")},
        "initial_capital": settings["backtest"]["initial_capital"],
    }


async def run_agent_cycle(
    agent,
    instruments: list[str],
    technical_snapshots: dict,   # instrument → TechnicalSnapshot
    market_data: dict,           # instrument → pl.DataFrame
    news: list,
    regime,
) -> list:
    """Run one signal-generation cycle for a single agent."""
    if not agent.is_active:
        return []
    # Filter to this agent's target instruments (if configured)
    targets = set(agent.target_instruments)
    filtered_tech = {k: v for k, v in technical_snapshots.items()
                     if not targets or k in targets}
    filtered_data = {k: v for k, v in market_data.items()
                     if not targets or k in targets}
    try:
        signals = await agent.generate_signals(filtered_data, filtered_tech, news, regime)
        return signals
    except Exception as exc:
        log.error("Agent signal error", agent=agent.agent_id, error=str(exc))
        return []


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Atlas Trader — Multi-Agent")
    parser.add_argument("--instruments", nargs="+",
                        default=["AAPL", "NVDA", "MSFT", "SPY", "EURUSD"],
                        help="Instruments to trade")
    parser.add_argument("--interval", type=int, default=900,
                        help="Cycle interval in seconds")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (no real orders)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()

    # Infrastructure
    notifier = Notifier(config["notifier"])
    store = DataStore(config["db_path"])
    data_fetcher = MarketDataFetcher(config["trading"], store)
    tech_engine = TechnicalEngine()
    claude_agent = ClaudeAgent(config["claude"])
    regime_detector = RegimeDetector()
    risk_mgr = RiskManager(config["hard_limits"], config["soft_limits"], store)
    position_sizer = PositionSizer(config["position_sizing"], config["hard_limits"], store)
    order_manager = OrderManager(config, store=store)
    journal = TradeJournal(store, claude_agent, notifier)
    attribution = AttributionEngine(store)

    initial_capital = config["initial_capital"]

    # Build agents
    agent_config = {"instruments": config["instruments"]}
    momentum = MomentumAgent(
        agent_id="momentum", config=agent_config,
        store=store, claude_agent=claude_agent, technical=tech_engine,
    )
    mean_rev = MeanReversionAgent(
        agent_id="mean_reversion", config=agent_config,
        store=store, claude_agent=claude_agent, technical=tech_engine,
    )
    event_driven = EventDrivenAgent(
        agent_id="event_driven", config=agent_config,
        store=store, claude_agent=claude_agent, technical=tech_engine,
    )
    stat_arb = StatArbAgent(
        agent_id="stat_arb", config=agent_config,
        store=store, claude_agent=claude_agent, technical=tech_engine,
    )
    agents = [momentum, mean_rev, event_driven, stat_arb]

    orchestrator = Orchestrator(agents, initial_capital, claude_agent)

    # Phase 3 portfolio-level components
    global_risk = GlobalRiskManager(config["hard_limits"])
    corr_monitor = CorrelationMonitor(agents)
    evolver = StrategyEvolver(agents, claude_agent, store, notifier)

    # Connect adapters (paper or live)
    if not args.paper:
        for instrument in args.instruments:
            try:
                adapter = AdapterFactory.get_adapter(instrument, config)
                await adapter.connect()
            except Exception as exc:
                log.warning("Adapter connect failed", instrument=instrument, error=str(exc))

    kill_switch = KillSwitch(
        max_drawdown_pct=config["hard_limits"]["max_drawdown_pct"],
        max_daily_loss_pct=config["hard_limits"]["max_daily_loss_pct"],
    )

    log.info("Multi-agent system started",
             agents=[a.agent_id for a in agents],
             instruments=args.instruments,
             paper=args.paper)
    await notifier.send(
        f"Multi-Agent System started\n"
        f"Agents: {', '.join(a.name for a in agents)}\n"
        f"Instruments: {', '.join(args.instruments)}"
    )

    cycle_count = 0
    last_perf_update = datetime.utcnow()
    last_weekly_review = datetime.utcnow()

    try:
        while not kill_switch.active:
            cycle_start = datetime.utcnow()

            # 1. Fetch market data and compute technicals
            technical_snapshots: dict = {}
            market_data: dict = {}
            regime = None

            for instrument in args.instruments:
                if not is_market_open(instrument):
                    continue
                try:
                    bars_15m = await data_fetcher.get_latest_bars(instrument, "15m", 300)
                    bars_1h = await data_fetcher.get_latest_bars(instrument, "1h", 300)
                    if bars_15m.is_empty():
                        continue
                    snap = tech_engine.compute_all(bars_15m, instrument, "15m")
                    technical_snapshots[instrument] = snap
                    market_data[instrument] = bars_15m
                    if regime is None and not bars_1h.is_empty():
                        regime = regime_detector.classify(bars_1h)
                except Exception as exc:
                    log.error("Data fetch error", instrument=instrument, error=str(exc))

            if not regime:
                from src.types import MarketRegime as MR
                regime = MR.RANGE_BOUND

            # 2. Allocate capital
            allocation = await orchestrator.allocate_capital(regime)

            # 3. Run all agents in parallel
            all_raw_signals = []
            agent_signal_tasks = [
                run_agent_cycle(agent, args.instruments, technical_snapshots,
                                market_data, [], regime)
                for agent in agents
            ]
            results = await asyncio.gather(*agent_signal_tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_raw_signals.extend(res)

            # 4. Conflict resolution
            resolved_signals = orchestrator.resolve_conflicts(all_raw_signals)

            log.info("Cycle signals",
                     raw=len(all_raw_signals), resolved=len(resolved_signals),
                     regime=regime.value)

            # 5. Execute resolved signals
            from src.types import PortfolioState, Position
            portfolio = PortfolioState(
                equity=initial_capital,
                cash=initial_capital,
                total_unrealized_pnl=0.0,
                total_realized_pnl=0.0,
            )

            for signal in resolved_signals:
                if signal.action in (TradeAction.NEUTRAL, TradeAction.NO_TRADE):
                    continue

                risk_result = risk_mgr.check_signal(signal, portfolio, [])
                if not risk_result.approved:
                    continue

                snap = technical_snapshots.get(signal.instrument)
                atr = snap.indicators.get("atr_14", 0.0) if snap else 0.0
                size = position_sizer.calculate(signal, portfolio, {}, atr)
                if size.units <= 0:
                    continue

                order_side = (Side.BUY if signal.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
                              else Side.SELL)
                order = Order(
                    id=f"MA-{uuid.uuid4().hex[:8]}",
                    instrument=signal.instrument,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    quantity=size.units,
                )

                if args.paper:
                    # Simulate fill at current price
                    close = (technical_snapshots[signal.instrument].indicators.get("close", 0)
                             if signal.instrument in technical_snapshots else 0)
                    log.info("Paper order",
                             instrument=signal.instrument, side=order_side.value,
                             qty=round(size.units, 4), price=close,
                             composite=round(signal.composite_score, 3))
                    store.save_signal(signal)
                else:
                    result = await order_manager.submit(order)
                    if result.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                        store.save_signal(signal)

            # 6. Update agent performance every 30 minutes
            if (datetime.utcnow() - last_perf_update).total_seconds() > 1800:
                for agent in agents:
                    agent.update_performance(initial_capital)
                # Correlation check
                corr_alerts = corr_monitor.check_alerts()
                for alert in corr_alerts:
                    log.warning("Correlation alert", msg=alert)
                div_score = corr_monitor.get_diversification_score()
                log.info("Portfolio diversification score", score=round(div_score, 3))
                last_perf_update = datetime.utcnow()

            # 7. Weekly strategy evolution (every 7 days)
            if (datetime.utcnow() - last_weekly_review).total_seconds() > 604800:
                try:
                    evolution_report = await evolver.weekly_review()
                    log.info("Weekly evolution complete",
                             agents_reviewed=len(evolution_report.get("agents", {})))
                except Exception as exc:
                    log.error("Weekly evolution failed", error=str(exc))
                last_weekly_review = datetime.utcnow()

            cycle_count += 1
            elapsed = (datetime.utcnow() - cycle_start).total_seconds()
            sleep_secs = max(0, args.interval - elapsed)
            log.info("Multi-agent cycle complete",
                     count=cycle_count, elapsed_s=round(elapsed, 1),
                     next_in_s=round(sleep_secs))
            await asyncio.sleep(sleep_secs)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        await AdapterFactory.disconnect_all()
        await data_fetcher.close()

        # Final weekly report
        from datetime import timedelta
        report = attribution.generate_report(
            datetime.utcnow() - timedelta(days=7), datetime.utcnow()
        )
        attribution.print_report(report)
        store.close()
        log.info("Multi-agent system stopped")


if __name__ == "__main__":
    asyncio.run(main())
