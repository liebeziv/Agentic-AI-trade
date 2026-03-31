"""Backtesting framework — walk-forward optimization over historical data."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml
from dotenv import load_dotenv

from src.analysis.regime_detector import RegimeDetector
from src.analysis.technical import TechnicalEngine
from src.data.data_store import DataStore
from src.data.market_data import MarketDataFetcher
from src.execution.paper_engine import PaperTradingEngine
from src.reflection.trade_journal import TradeJournal
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.strategy.signal_aggregator import SignalAggregator
from src.types import (
    Bar, MarketRegime, Order, OrderType, PortfolioState,
    SignalScore, Side, TradeAction, TradeRecord,
)
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import (
    calmar_ratio, max_drawdown, profit_factor, sharpe_ratio, win_rate,
)

load_dotenv()
log = get_logger(__name__)


@dataclass
class BacktestResults:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    avg_trade_duration_min: float = 0.0
    avg_risk_reward_achieved: float = 0.0
    monthly_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[TradeRecord] = field(default_factory=list)


def compute_results(
    trades: list[TradeRecord], equity_curve: list[float], initial_capital: float
) -> BacktestResults:
    if not trades:
        return BacktestResults()

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]

    total_pnl = sum(pnls)
    total_pnl_pct = total_pnl / initial_capital * 100

    # Annualised Sharpe (assume 252 trading days, return per trade)
    daily_returns = [t.pnl / initial_capital for t in trades]
    sh = sharpe_ratio(daily_returns)

    # Sortino (downside only)
    import numpy as np
    neg = [r for r in daily_returns if r < 0]
    downside_std = float(np.std(neg)) if neg else 1e-9
    sortino = float(np.mean(daily_returns)) / downside_std * (252 ** 0.5) if downside_std > 0 else 0

    md = max_drawdown(equity_curve) if equity_curve else 0.0
    annual_ret = total_pnl_pct / max(len(trades), 1) * 252  # rough annualisation
    calmar = calmar_ratio(annual_ret, md) if md > 0 else 0.0
    pf = profit_factor(pnls)
    wr = win_rate(pnls)

    return BacktestResults(
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=wr,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
        avg_win_pct=float(np.mean([t.pnl_pct for t in wins])) if wins else 0,
        avg_loss_pct=float(np.mean([t.pnl_pct for t in losses])) if losses else 0,
        profit_factor=pf,
        sharpe_ratio=sh,
        sortino_ratio=sortino,
        max_drawdown_pct=md,
        calmar_ratio=calmar,
        avg_trade_duration_min=float(np.mean([t.duration_minutes for t in trades])),
        equity_curve=equity_curve,
        trade_log=trades,
    )


class RuleBasedClaudeMock:
    """Lightweight rule-based mock for Claude during backtesting (no API calls)."""

    class CostTracker:
        estimated_cost = 0.0
        def can_make_call(self, *a, **kw): return True

    cost_tracker = CostTracker()

    async def analyze_market(self, instrument, technical, news, regime, recent_trades, portfolio_state):
        from src.types import TradeAction, TradeRecommendation
        # Use primary timeframe (prefer 15m, else first available)
        snap = technical.get("15m") or technical.get("1h") or next(iter(technical.values()), None)
        if snap is None or not snap.indicators:
            return None

        close = snap.indicators.get("close", 0) or 1
        atr   = snap.indicators.get("atr_14", close * 0.005) or close * 0.005
        rsi   = snap.indicators.get("rsi_14", 50)
        macd_hist = snap.indicators.get("macd_hist", 0)
        adx   = snap.indicators.get("adx_14", 15)
        sma20 = snap.indicators.get("sma_20", close)
        sma50 = snap.indicators.get("sma_50", close)

        # Score: +1 = strong buy, -1 = strong sell
        score = 0.0
        if rsi < 40: score += 0.4
        elif rsi > 60: score -= 0.4
        if macd_hist > 0: score += 0.3
        elif macd_hist < 0: score -= 0.3
        if sma20 > sma50: score += 0.3
        elif sma20 < sma50: score -= 0.3

        # Only trade when score is decisive and trend is present
        if abs(score) < 0.5 or adx < 15:
            return None

        if score > 0:
            action = TradeAction.BUY
            confidence = min(90, int(50 + score * 40))
            sl = close - atr * 2
            tp = close + atr * 3
        else:
            action = TradeAction.SELL
            confidence = min(90, int(50 + abs(score) * 40))
            sl = close + atr * 2
            tp = close - atr * 3

        rr = abs(tp - close) / abs(sl - close) if abs(sl - close) > 0 else 1.5

        return TradeRecommendation(
            action=action, instrument=instrument, confidence=confidence,
            entry_price=close, stop_loss=sl, take_profit_1=tp,
            risk_reward_ratio=round(rr, 2), reasoning="Rule-based mock (RSI+MACD+MA)",
        )

    async def reflect_on_trade(self, *a, **kw):
        return {}


async def run_backtest(
    instrument: str,
    start: datetime,
    end: datetime,
    config: dict,
    use_claude: bool = False,
) -> BacktestResults:
    """Run a single backtest window for one instrument."""
    store = DataStore(":memory:")
    data_fetcher = MarketDataFetcher(config["trading"], store)
    technical = TechnicalEngine()
    claude_agent = RuleBasedClaudeMock() if not use_claude else None
    regime_detector = RegimeDetector()
    signal_agg = SignalAggregator()
    risk_mgr = RiskManager(config["hard_limits"], config["soft_limits"], store)
    position_sizer = PositionSizer(config["position_sizing"], config["hard_limits"], store)
    paper_engine = PaperTradingEngine(config["paper"], store)
    journal = TradeJournal(store, claude_agent)

    initial_capital = config["paper"]["initial_capital"]

    # yfinance limits: 15m → last 60 days, 1h → last 730 days
    from datetime import timedelta
    now = datetime.utcnow()
    limit_15m = now - timedelta(days=59)
    start_15m = max(start, limit_15m)

    # Fetch historical data once
    log.info("Fetching historical data", instrument=instrument, start=start, end=end)
    bars_1h = await data_fetcher.get_historical(instrument, "1h", start, end)
    bars_15m = await data_fetcher.get_historical(instrument, "15m", start_15m, end)
    bars_5m = await data_fetcher.get_historical(instrument, "5m", start_15m, end)

    # Fall back to 1h if 15m unavailable
    if bars_15m.is_empty():
        if bars_1h.is_empty():
            log.warning("No data available", instrument=instrument)
            return BacktestResults()
        log.warning("15m unavailable, falling back to 1h bars", instrument=instrument)
        bars_15m = bars_1h

    equity_curve: list[float] = [initial_capital]
    all_trades: list[TradeRecord] = []

    # Replay 15m bars
    warmup = 200  # bars needed for indicators
    bars_list = bars_15m.to_dicts()

    for i in range(warmup, len(bars_list)):
        window = pl.DataFrame(bars_list[:i+1])
        current_row = bars_list[i]
        current_bar = Bar(
            timestamp=current_row["timestamp"],
            open=current_row["open"], high=current_row["high"],
            low=current_row["low"], close=current_row["close"],
            volume=current_row["volume"], instrument=instrument, timeframe="15m",
        )

        # Technical snapshot from window
        snap_15m = technical.compute_all(window, instrument, "15m")
        technical_snapshots = {"15m": snap_15m}

        # Regime from 1h window
        cutoff = current_row["timestamp"]
        h1_window = bars_1h.filter(pl.col("timestamp") <= cutoff).tail(200)
        regime = regime_detector.classify(h1_window) if not h1_window.is_empty() else MarketRegime.RANGE_BOUND

        # Mock Claude
        rec = await claude_agent.analyze_market(
            instrument=instrument, technical=technical_snapshots,
            news=[], regime=regime,
            recent_trades=store.get_trades(limit=10),
            portfolio_state=paper_engine.portfolio,
        )

        # Build signal
        claude_conf = (rec.confidence / 100) if rec else 0.5
        sentiment = 0.0
        if rec:
            if rec.action in (TradeAction.BUY, TradeAction.STRONG_BUY):
                sentiment = claude_conf
            elif rec.action in (TradeAction.SELL, TradeAction.STRONG_SELL):
                sentiment = -claude_conf

        signal = signal_agg.build_signal(
            instrument=instrument, technical_snap=snap_15m,
            regime=regime, claude_confidence=claude_conf,
            sentiment_score=sentiment, recommendation=rec,
        )

        if signal.action in (TradeAction.NEUTRAL, TradeAction.NO_TRADE):
            pass
        else:
            risk_result = risk_mgr.check_signal(signal, paper_engine.portfolio, [])
            if risk_result.approved:
                atr = snap_15m.indicators.get("atr_14", 0.0)
                size = position_sizer.calculate(signal, paper_engine.portfolio, {}, atr)
                if risk_result.adjusted_size:
                    size.units *= risk_result.adjusted_size

                if size.units > 0:
                    order_side = (
                        Side.BUY if signal.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
                        else Side.SELL
                    )
                    import uuid as _uuid
                    order = Order(
                        id=f"ORD-{_uuid.uuid4().hex[:8]}",
                        instrument=instrument, side=order_side,
                        order_type=OrderType.MARKET, quantity=size.units,
                        stop_loss=rec.stop_loss if rec else None,
                        take_profit=rec.take_profit_1 if rec else None,
                    )
                    await paper_engine.execute_order(order, current_bar)
                    risk_mgr.record_trade_executed(instrument)

        # Check stops
        closed = paper_engine.check_stops(current_bar)
        all_trades.extend(closed)
        paper_engine.update_mark_to_market({instrument: current_bar.close})
        equity_curve.append(paper_engine.portfolio.equity)

    # Force close remaining positions at end
    if bars_list:
        last_price = bars_list[-1]["close"]
        closed = paper_engine.force_close_all({instrument: last_price})
        all_trades.extend(closed)

    return compute_results(all_trades, equity_curve, initial_capital)


async def walk_forward_backtest(
    instrument: str,
    config: dict,
    start_date: datetime,
    end_date: datetime,
    train_days: int = 90,
    test_days: int = 30,
) -> list[tuple[datetime, datetime, BacktestResults]]:
    results = []
    current = start_date + timedelta(days=train_days)

    while current + timedelta(days=test_days) <= end_date:
        test_start = current
        test_end = current + timedelta(days=test_days)

        log.info("Walk-forward window", instrument=instrument,
                 test_start=test_start.date(), test_end=test_end.date())

        result = await run_backtest(instrument, test_start, test_end, config)
        results.append((test_start, test_end, result))

        current += timedelta(days=test_days)

    return results


def print_results(results: BacktestResults, instrument: str) -> None:
    print(f"\n{'='*55}")
    print(f"  BACKTEST RESULTS — {instrument}")
    print(f"{'='*55}")
    print(f"  Total Trades:    {results.total_trades:>8d}")
    print(f"  Win Rate:        {results.win_rate:>7.1f}%")
    print(f"  Total P&L:       ${results.total_pnl:>10.2f}  ({results.total_pnl_pct:.1f}%)")
    print(f"  Avg Win:         {results.avg_win_pct:>7.2f}%")
    print(f"  Avg Loss:        {results.avg_loss_pct:>7.2f}%")
    print(f"  Profit Factor:   {results.profit_factor:>9.2f}")
    print(f"  Sharpe Ratio:    {results.sharpe_ratio:>9.2f}")
    print(f"  Max Drawdown:    {results.max_drawdown_pct:>7.1f}%")
    print(f"  Calmar Ratio:    {results.calmar_ratio:>9.2f}")
    print(f"  Avg Duration:    {results.avg_trade_duration_min:>6.0f} min")
    print(f"{'='*55}\n")


def load_config() -> dict:
    base = Path(__file__).parent.parent
    with open(base / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(base / "config" / "risk_params.yaml") as f:
        risk_cfg = yaml.safe_load(f)

    return {
        "trading": settings["trading"],
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Atlas Trader Backtester")
    parser.add_argument("--instrument", default="AAPL", help="Instrument symbol")
    parser.add_argument("--days", type=int, default=90, help="Lookback days")
    parser.add_argument("--walk-forward", action="store_true", help="Walk-forward mode")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config()

    end = datetime.utcnow()
    start = end - timedelta(days=args.days)

    if args.walk_forward:
        windows = asyncio.run(walk_forward_backtest(
            args.instrument, config, start, end,
            train_days=60, test_days=30,
        ))
        for s, e, r in windows:
            print(f"\nWindow {s.date()} → {e.date()}")
            print_results(r, args.instrument)
    else:
        results = asyncio.run(run_backtest(args.instrument, start, end, config))
        print_results(results, args.instrument)


if __name__ == "__main__":
    main()
