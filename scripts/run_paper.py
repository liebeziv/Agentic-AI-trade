"""Main paper trading loop — connects all Phase 1 modules."""
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
from src.data.economic_calendar import EconomicCalendar
from src.data.market_data import MarketDataFetcher
from src.data.news_feed import NewsFeed
from src.execution.paper_engine import PaperTradingEngine
from src.reflection.trade_journal import TradeJournal
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.strategy.signal_aggregator import SignalAggregator
from src.types import Order, OrderType, Side, TradeAction
from src.utils.logger import get_logger, setup_logging
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
        "backtest": settings.get("backtest", {}),
        "instruments": instruments_cfg["instruments"],
        "hard_limits": risk_cfg["hard_limits"],
        "soft_limits": risk_cfg["soft_limits"],
        "position_sizing": risk_cfg["position_sizing"],
        "db_path": os.getenv("DUCKDB_PATH", "./data/atlas.duckdb"),
        "news": {
            "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
            "fetch_interval_seconds": 300,
        },
        "paper": {
            "initial_capital": settings["backtest"]["initial_capital"],
            "commission_pct": settings["backtest"]["commission_pct"],
            "slippage_mean_pct": settings["backtest"]["slippage_pct"],
            "slippage_std_pct": settings["backtest"]["slippage_pct"] / 2,
            "latency_ms": 50,
        },
    }


def get_all_symbols(instruments_cfg: dict) -> list[str]:
    symbols: list[str] = []
    for category in instruments_cfg.values():
        for item in category:
            symbols.append(item["symbol"])
    return symbols


async def trading_loop(config: dict) -> None:
    setup_logging(
        log_level=config["system"]["log_level"],
        log_dir=config["system"]["log_dir"],
    )

    # --- Initialise all modules ---
    store = DataStore(config["db_path"])
    data_fetcher = MarketDataFetcher(config["trading"], store)
    technical = TechnicalEngine()
    claude_agent = ClaudeAgent(config["claude"])
    regime_detector = RegimeDetector()
    signal_agg = SignalAggregator()
    risk_mgr = RiskManager(config["hard_limits"], config["soft_limits"], store)
    position_sizer = PositionSizer(config["position_sizing"], config["hard_limits"], store)
    paper_engine = PaperTradingEngine(config["paper"], store)
    journal = TradeJournal(store, claude_agent)
    news_feed = NewsFeed(config["news"])
    calendar = EconomicCalendar()

    # Build instrument list (limit to top instruments for paper trading)
    instruments = config["trading"].get("instruments", get_all_symbols(config["instruments"])[:5])
    interval = config["trading"]["analysis_interval_seconds"]

    log.info("Atlas Trader starting", mode="paper", instruments=instruments)

    # Initial calendar fetch
    await calendar.refresh()

    loop_count = 0

    while True:
        cycle_start = datetime.utcnow()
        loop_count += 1
        log.info("Cycle start", n=loop_count, equity=round(paper_engine.portfolio.equity, 2))

        # Refresh calendar every 12 hours
        if calendar.needs_refresh():
            await calendar.refresh()

        # Fetch news for all instruments
        try:
            fresh_news = await news_feed.fetch_all(instruments)
            for item in fresh_news:
                news_feed._cache[f"{item.timestamp.isoformat()}_{hash(item.title)}"] = item
        except Exception as exc:
            log.warning("News fetch failed", error=str(exc))

        # --- Per-instrument analysis loop ---
        for instrument in instruments:
            instr_type = data_fetcher.get_instrument_type(instrument)
            if not is_market_open(instr_type):
                log.debug("Market closed, skipping", instrument=instrument)
                continue

            try:
                # 1. PERCEIVE — Multi-timeframe technical analysis
                technical_snapshots = await technical.multi_timeframe_analysis(
                    instrument, data_fetcher, ["5m", "15m", "1h"]
                )

                # Use 15m as primary timeframe for signal
                primary_snap = technical_snapshots.get("15m") or next(iter(technical_snapshots.values()))
                if primary_snap is None or not primary_snap.indicators:
                    log.debug("Insufficient data", instrument=instrument)
                    continue

                # News for this instrument
                news = await news_feed.get_recent(instrument, hours=4)

                # Economic calendar events
                events = calendar.upcoming_events(hours=4)

                # 2. ANALYZE — Regime + Claude
                regime = regime_detector.classify(
                    await data_fetcher.get_latest_bars(instrument, "1h", count=200)
                )

                recommendation = None
                sentiment_score = 0.0
                claude_confidence = 0.5

                if claude_agent.cost_tracker.can_make_call():
                    recommendation = await claude_agent.analyze_market(
                        instrument=instrument,
                        technical=technical_snapshots,
                        news=news,
                        regime=regime,
                        recent_trades=store.get_trades(limit=10),
                        portfolio_state=paper_engine.portfolio,
                    )

                if recommendation:
                    claude_confidence = recommendation.confidence / 100
                    if recommendation.action in (TradeAction.BUY, TradeAction.STRONG_BUY):
                        sentiment_score = claude_confidence
                    elif recommendation.action in (TradeAction.SELL, TradeAction.STRONG_SELL):
                        sentiment_score = -claude_confidence

                # 3. DECIDE — Signal aggregation
                signal = signal_agg.build_signal(
                    instrument=instrument,
                    technical_snap=primary_snap,
                    regime=regime,
                    claude_confidence=claude_confidence,
                    sentiment_score=sentiment_score,
                    recommendation=recommendation,
                )
                store.save_signal(signal)

                if signal.action in (TradeAction.NEUTRAL, TradeAction.NO_TRADE):
                    continue

                # 4. RISK CHECK
                risk_result = risk_mgr.check_signal(signal, paper_engine.portfolio, events)
                if not risk_result.approved:
                    log.info("Signal vetoed by risk manager",
                             instrument=instrument, reason=risk_result.reason)
                    continue

                # 5. SIZE POSITION
                atr = primary_snap.indicators.get("atr_14", 0.0)
                size = position_sizer.calculate(signal, paper_engine.portfolio, {}, atr)

                if risk_result.adjusted_size:
                    size.units *= risk_result.adjusted_size
                    size.risk_amount *= risk_result.adjusted_size

                if size.units <= 0:
                    continue

                # 6. EXECUTE (paper)
                latest_bars = await data_fetcher.get_latest_bars(instrument, "5m", count=1)
                if latest_bars.is_empty():
                    continue
                row = latest_bars.row(-1, named=True)

                from src.types import Bar
                current_bar = Bar(
                    timestamp=row["timestamp"],
                    open=row["open"], high=row["high"],
                    low=row["low"], close=row["close"],
                    volume=row["volume"],
                    instrument=instrument, timeframe="5m",
                )

                order_side = (
                    Side.BUY
                    if signal.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
                    else Side.SELL
                )
                order = Order(
                    id=f"ORD-{uuid.uuid4().hex[:8]}",
                    instrument=instrument,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    quantity=size.units,
                    stop_loss=recommendation.stop_loss if recommendation else None,
                    take_profit=recommendation.take_profit_1 if recommendation else None,
                    signal_id=f"SIG-{signal.timestamp.strftime('%H%M%S')}",
                    claude_reasoning=recommendation.reasoning if recommendation else "",
                )

                filled_order = await paper_engine.execute_order(order, current_bar)
                if filled_order.status.value == "FILLED":
                    risk_mgr.record_trade_executed(instrument)
                    log.info("Order filled", id=filled_order.id,
                             instrument=instrument, side=order_side.value,
                             qty=size.units, price=filled_order.fill_price)

            except Exception as exc:
                log.error("Error processing instrument", instrument=instrument, error=str(exc))

        # --- Stop check for all open positions ---
        all_prices: dict[str, float] = {}
        for pos in paper_engine.portfolio.positions[:]:
            try:
                bars = await data_fetcher.get_latest_bars(pos.instrument, "5m", count=1)
                if not bars.is_empty():
                    price = float(bars["close"][-1])
                    all_prices[pos.instrument] = price

                    from src.types import Bar
                    bar = Bar(
                        timestamp=bars["timestamp"][-1],
                        open=float(bars["open"][-1]),
                        high=float(bars["high"][-1]),
                        low=float(bars["low"][-1]),
                        close=price,
                        volume=float(bars["volume"][-1]),
                        instrument=pos.instrument, timeframe="5m",
                    )
                    closed = paper_engine.check_stops(bar)
                    for trade in closed:
                        await journal.record_trade(trade, paper_engine.portfolio)

            except Exception as exc:
                log.warning("Stop check error", instrument=pos.instrument, error=str(exc))

        # --- Mark-to-market update ---
        if all_prices:
            paper_engine.update_mark_to_market(all_prices)

        # --- Portfolio snapshot ---
        store.save_portfolio_snapshot(paper_engine.portfolio)

        log.info("Cycle complete",
                 equity=round(paper_engine.portfolio.equity, 2),
                 positions=len(paper_engine.portfolio.positions),
                 daily_pnl=round(paper_engine.portfolio.daily_pnl, 2),
                 claude_cost_usd=round(claude_agent.cost_tracker.estimated_cost, 4))

        # Wait for next cycle
        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        sleep_time = max(0, interval - elapsed)
        await asyncio.sleep(sleep_time)


def main() -> None:
    config = load_config()
    # Default instruments for paper trading (configurable via settings.yaml)
    if "instruments" not in config["trading"]:
        config["trading"]["instruments"] = ["EURUSD", "BTC/USDT", "AAPL", "SPY", "GC"]
    asyncio.run(trading_loop(config))


if __name__ == "__main__":
    main()
