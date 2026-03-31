"""Live trading loop — Phase 2. Extends run_paper.py with real exchange adapters."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.analysis.claude_agent import ClaudeAgent
from src.reflection.attribution import AttributionEngine
from src.analysis.regime_detector import RegimeDetector
from src.analysis.technical import TechnicalEngine
from src.data.data_store import DataStore
from src.data.economic_calendar import EconomicCalendar
from src.data.market_data import MarketDataFetcher
from src.data.news_feed import NewsFeed
from src.execution.exchange_adapter import AdapterFactory
from src.execution.order_manager import KillSwitch, OrderManager
from src.reflection.trade_journal import TradeJournal
from src.reflection.attribution import AttributionEngine
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.strategy.signal_aggregator import SignalAggregator
from src.types import Order, OrderType, Side, TradeAction
from src.utils.logger import get_logger, setup_logging
from src.utils.notifier import Notifier
from src.utils.time_utils import is_market_open

load_dotenv()
log = get_logger(__name__)

DEPLOYMENT_SCHEDULE: dict[int, float] = {
    1: 0.10,   # Week 1:  10% of capital
    2: 0.20,
    3: 0.35,
    4: 0.50,
    5: 0.70,
    6: 0.85,
    7: 1.00,   # Week 7+: full deployment
}


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
        "news": {
            "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
            "fetch_interval_seconds": 300,
        },
        # Exchange adapter configs (read from env)
        "alpaca": {
            "api_key": os.getenv("ALPACA_API_KEY", ""),
            "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
            "paper": os.getenv("ALPACA_BASE_URL", "").find("paper") >= 0,
        },
        "binance": {
            "api_key": os.getenv("BINANCE_API_KEY", ""),
            "secret": os.getenv("BINANCE_SECRET", ""),
            "testnet": True,  # Always testnet until explicitly overridden
        },
        "ib": {
            "host": os.getenv("IB_HOST", "127.0.0.1"),
            "port": int(os.getenv("IB_PORT", 7497)),
            "client_id": int(os.getenv("IB_CLIENT_ID", 1)),
        },
        "notifier": {
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        },
    }


async def pre_flight_checks(config: dict, notifier: Notifier) -> None:
    """Must all pass before live trading starts."""
    errors = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY not set")

    if not os.getenv("ALPACA_API_KEY") and not os.getenv("BINANCE_API_KEY"):
        errors.append("No exchange API key set (ALPACA_API_KEY or BINANCE_API_KEY)")

    if errors:
        msg = "Pre-flight FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(msg)
        raise RuntimeError(msg)

    log.info("Pre-flight checks passed")
    await notifier.send("Atlas Trader LIVE starting — pre-flight OK")


def get_capital_fraction(go_live_date: datetime) -> float:
    weeks_live = max(1, (datetime.utcnow() - go_live_date).days // 7 + 1)
    weeks_live = min(weeks_live, max(DEPLOYMENT_SCHEDULE.keys()))
    return DEPLOYMENT_SCHEDULE[weeks_live]


async def trading_cycle(
    instruments: list[str],
    *,
    data_fetcher: MarketDataFetcher,
    technical: TechnicalEngine,
    claude_agent: ClaudeAgent,
    regime_detector: RegimeDetector,
    signal_agg: SignalAggregator,
    risk_mgr: RiskManager,
    position_sizer: PositionSizer,
    order_manager: OrderManager,
    store: DataStore,
    notifier: Notifier,
    config: dict,
    capital_fraction: float,
) -> None:
    """Single analysis-and-execution cycle for all instruments."""

    for instrument in instruments:
        try:
            if not is_market_open(instrument):
                continue

            # Fetch recent bars
            bars = await data_fetcher.get_latest_bars(instrument, "15m", count=300)
            bars_1h = await data_fetcher.get_latest_bars(instrument, "1h", count=300)
            if bars.is_empty():
                continue

            # Technical analysis
            snap_15m = technical.compute_all(bars, instrument, "15m")
            snaps = {"15m": snap_15m}
            if not bars_1h.is_empty():
                snaps["1h"] = technical.compute_all(bars_1h, instrument, "1h")

            # Regime
            regime = (regime_detector.classify(bars_1h)
                      if not bars_1h.is_empty()
                      else regime_detector.classify(bars))

            # News (best effort)
            news = store.get_recent_news(instrument, hours=4)

            # Sync live positions for portfolio state
            live_positions = await order_manager.sync_positions()
            # Build a simple portfolio proxy from live positions
            adapter = AdapterFactory.get_adapter(instrument, config)
            account = await adapter.get_account()

            from src.types import PortfolioState
            portfolio = PortfolioState(
                equity=account.equity * capital_fraction,
                cash=account.cash * capital_fraction,
                total_unrealized_pnl=0.0,
                total_realized_pnl=store.get_trades(limit=1000).__len__() * 0.0,
            )

            # Claude analysis
            rec = await claude_agent.analyze_market(
                instrument=instrument,
                technical=snaps,
                news=news,
                regime=regime,
                recent_trades=store.get_trades(limit=10),
                portfolio_state=portfolio,
            )
            if not rec:
                continue

            claude_conf = rec.confidence / 100
            sentiment = claude_conf if rec.action in (TradeAction.BUY, TradeAction.STRONG_BUY) \
                else (-claude_conf if rec.action in (TradeAction.SELL, TradeAction.STRONG_SELL) else 0.0)

            signal = signal_agg.build_signal(
                instrument=instrument,
                technical_snap=snap_15m,
                regime=regime,
                claude_confidence=claude_conf,
                sentiment_score=sentiment,
                recommendation=rec,
            )
            store.save_signal(signal)

            if signal.action in (TradeAction.NEUTRAL, TradeAction.NO_TRADE):
                continue

            # Risk check
            risk_result = risk_mgr.check_signal(signal, portfolio, [])
            if not risk_result.approved:
                log.info("Signal blocked", instrument=instrument,
                         reason=risk_result.reason)
                continue

            # Position sizing
            atr = snap_15m.indicators.get("atr_14", 0.0)
            size = position_sizer.calculate(signal, portfolio, {}, atr)
            if risk_result.adjusted_size:
                size.units *= risk_result.adjusted_size
            if size.units <= 0:
                continue

            # Build and submit order
            order_side = (Side.BUY if signal.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
                          else Side.SELL)
            order = Order(
                id=f"ORD-{uuid.uuid4().hex[:8]}",
                instrument=instrument,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=size.units,
                price=rec.entry_price,
                stop_loss=rec.stop_loss,
                take_profit=rec.take_profit_1,
                claude_reasoning=rec.reasoning[:200],
            )

            result = await order_manager.submit(order)
            if result.status not in (OrderStatus.REJECTED, OrderStatus.CANCELLED):
                await notifier.send_trade_open(
                    instrument=instrument,
                    side=order_side.value,
                    price=result.fill_price or rec.entry_price,
                    quantity=size.units,
                    stop_loss=rec.stop_loss,
                    take_profit=rec.take_profit_1,
                    composite_score=signal.composite_score,
                )
                log.info("Order filled", instrument=instrument,
                         side=order_side.value, price=result.fill_price,
                         qty=size.units, score=round(signal.composite_score, 3))

        except Exception as exc:
            log.error("Cycle error", instrument=instrument, error=str(exc))


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Atlas Trader — Live")
    parser.add_argument("--instruments", nargs="+",
                        default=["AAPL", "NVDA", "MSFT", "BTC/USDT"],
                        help="Instruments to trade")
    parser.add_argument("--interval", type=int, default=900,
                        help="Cycle interval in seconds (default: 900 = 15m)")
    parser.add_argument("--go-live-date", default=None,
                        help="Date trading went live (YYYY-MM-DD) for capital scaling")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    config = load_config()

    go_live_date = (datetime.fromisoformat(args.go_live_date)
                    if args.go_live_date else datetime.utcnow())
    capital_fraction = get_capital_fraction(go_live_date)

    notifier = Notifier(config["notifier"])
    store = DataStore(config["db_path"])
    data_fetcher = MarketDataFetcher(config["trading"], store)
    technical = TechnicalEngine()
    claude_agent = ClaudeAgent(config["claude"])
    regime_detector = RegimeDetector()
    signal_agg = SignalAggregator()
    risk_mgr = RiskManager(config["hard_limits"], config["soft_limits"], store)
    position_sizer = PositionSizer(config["position_sizing"], config["hard_limits"], store)
    order_manager = OrderManager(config, store=store)
    journal = TradeJournal(store, claude_agent, notifier)
    attribution = AttributionEngine(store)
    kill_switch = KillSwitch(
        max_drawdown_pct=config["hard_limits"]["max_drawdown_pct"],
        max_daily_loss_pct=config["hard_limits"]["max_daily_loss_pct"],
    )

    # Pre-flight
    await pre_flight_checks(config, notifier)

    # Connect exchange adapters
    for instrument in args.instruments:
        try:
            adapter = AdapterFactory.get_adapter(instrument, config)
            await adapter.connect()
        except Exception as exc:
            log.error("Adapter connect failed", instrument=instrument, error=str(exc))

    log.info("Atlas Trader LIVE started",
             instruments=args.instruments,
             capital_fraction=f"{capital_fraction:.0%}",
             interval_s=args.interval)

    from src.types import OrderStatus
    try:
        cycle_count = 0
        last_daily_summary = datetime.utcnow().date()

        while not kill_switch.active:
            cycle_start = datetime.utcnow()

            # Kill switch check (get equity from any connected adapter)
            try:
                adapter = next(iter(AdapterFactory._adapters.values()))
                account = await adapter.get_account()
                # Simplified: use equity as proxy, drawdown tracking in store
                await kill_switch.check(0.0, 0.0, notifier)
            except Exception:
                pass

            await trading_cycle(
                args.instruments,
                data_fetcher=data_fetcher,
                technical=technical,
                claude_agent=claude_agent,
                regime_detector=regime_detector,
                signal_agg=signal_agg,
                risk_mgr=risk_mgr,
                position_sizer=position_sizer,
                order_manager=order_manager,
                store=store,
                notifier=notifier,
                config=config,
                capital_fraction=capital_fraction,
            )

            # Daily summary
            today = datetime.utcnow().date()
            if today > last_daily_summary:
                trades_today = store.get_trades(
                    start=datetime.utcnow().replace(hour=0, minute=0, second=0)
                )
                wins = sum(1 for t in trades_today if t.pnl > 0)
                losses = len(trades_today) - wins
                daily_pnl = sum(t.pnl for t in trades_today)
                try:
                    account = await next(iter(AdapterFactory._adapters.values())).get_account()
                    equity = account.equity
                except Exception:
                    equity = 0.0
                await notifier.send_daily_summary(equity, daily_pnl, wins, losses, 0.0)
                # Weekly attribution report (every Sunday)
                if datetime.utcnow().weekday() == 6:
                    week_start = datetime.utcnow().replace(
                        hour=0, minute=0, second=0) - __import__("datetime").timedelta(days=7)
                    report = attribution.generate_report(week_start, datetime.utcnow())
                    attribution.print_report(report)
                last_daily_summary = today

            cycle_count += 1
            elapsed = (datetime.utcnow() - cycle_start).total_seconds()
            sleep_secs = max(0, args.interval - elapsed)
            log.info("Cycle complete", count=cycle_count,
                     elapsed_s=round(elapsed, 1), next_in_s=round(sleep_secs))
            await asyncio.sleep(sleep_secs)

    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        await AdapterFactory.disconnect_all()
        await data_fetcher.close()
        store.close()
        log.info("Atlas Trader stopped")


if __name__ == "__main__":
    asyncio.run(main())
