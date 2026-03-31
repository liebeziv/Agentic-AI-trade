"""Tests for the paper trading engine."""
import asyncio
import pytest
from datetime import datetime

from src.execution.paper_engine import PaperTradingEngine
from src.types import Bar, ExitReason, Order, OrderStatus, OrderType, Side

CONFIG = {
    "initial_capital": 100_000.0,
    "commission_pct": 0.1,
    "slippage_mean_pct": 0.02,
    "slippage_std_pct": 0.01,
    "latency_ms": 0,
}


def make_bar(close: float = 100.0, instrument: str = "TEST") -> Bar:
    return Bar(
        timestamp=datetime.utcnow(),
        open=close * 0.999, high=close * 1.005,
        low=close * 0.995, close=close,
        volume=10000, instrument=instrument, timeframe="5m",
    )


def make_order(side: Side = Side.BUY, qty: float = 10.0, sl: float | None = None,
               tp: float | None = None) -> Order:
    return Order(
        id="ORD-TEST-001",
        instrument="TEST", side=side,
        order_type=OrderType.MARKET, quantity=qty,
        stop_loss=sl, take_profit=tp,
    )


def test_market_buy_fills():
    engine = PaperTradingEngine(CONFIG)
    bar = make_bar(100.0)
    order = make_order(Side.BUY, 10.0)
    result = asyncio.run(engine.execute_order(order, bar))
    assert result.status == OrderStatus.FILLED
    assert result.fill_price is not None
    assert result.fill_price >= 100.0  # slippage adds cost for buys


def test_market_sell_fills():
    engine = PaperTradingEngine(CONFIG)
    bar = make_bar(100.0)
    order = make_order(Side.SELL, 10.0)
    result = asyncio.run(engine.execute_order(order, bar))
    assert result.status == OrderStatus.FILLED
    assert result.fill_price is not None
    assert result.fill_price <= 100.0


def test_position_opens_on_fill():
    engine = PaperTradingEngine(CONFIG)
    bar = make_bar(100.0)
    order = make_order(Side.BUY, 5.0)
    asyncio.run(engine.execute_order(order, bar))
    assert len(engine.portfolio.positions) == 1
    assert engine.portfolio.positions[0].instrument == "TEST"


def test_stop_loss_triggered():
    engine = PaperTradingEngine(CONFIG)
    bar_open = make_bar(100.0)
    order = make_order(Side.BUY, 10.0, sl=95.0)
    asyncio.run(engine.execute_order(order, bar_open))

    # Bar where price drops below stop
    bar_sl = make_bar(94.0)
    bar_sl = Bar(
        timestamp=datetime.utcnow(), open=96.0, high=96.5,
        low=93.0, close=94.0, volume=10000,
        instrument="TEST", timeframe="5m",
    )
    closed = engine.check_stops(bar_sl)
    assert len(closed) == 1
    assert closed[0].exit_reason == ExitReason.STOP_LOSS


def test_take_profit_triggered():
    engine = PaperTradingEngine(CONFIG)
    bar_open = make_bar(100.0)
    order = make_order(Side.BUY, 10.0, sl=95.0, tp=110.0)
    asyncio.run(engine.execute_order(order, bar_open))

    bar_tp = Bar(
        timestamp=datetime.utcnow(), open=105.0, high=115.0,
        low=104.0, close=112.0, volume=10000,
        instrument="TEST", timeframe="5m",
    )
    closed = engine.check_stops(bar_tp)
    assert len(closed) == 1
    assert closed[0].exit_reason == ExitReason.TAKE_PROFIT
    assert closed[0].pnl > 0


def test_mark_to_market_updates_equity():
    engine = PaperTradingEngine(CONFIG)
    bar = make_bar(100.0)
    order = make_order(Side.BUY, 10.0, sl=95.0)
    asyncio.run(engine.execute_order(order, bar))

    # Record equity right after fill (commission already deducted)
    equity_after_fill = engine.portfolio.equity

    # Price goes up 10% — unrealized PnL should lift equity above post-fill level
    engine.update_mark_to_market({"TEST": 110.0})
    assert engine.portfolio.equity > equity_after_fill


def test_limit_order_pending_if_not_reached():
    engine = PaperTradingEngine(CONFIG)
    bar = make_bar(100.0)
    order = Order(
        id="ORD-LMT-001", instrument="TEST",
        side=Side.BUY, order_type=OrderType.LIMIT,
        quantity=10.0, price=90.0,  # limit below current price
    )
    result = asyncio.run(engine.execute_order(order, bar))
    assert result.status == OrderStatus.PENDING
    assert len(engine.portfolio.positions) == 0
