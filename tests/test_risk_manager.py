"""Tests for the risk manager."""
import pytest
from datetime import datetime

from src.strategy.risk_manager import RiskManager
from src.types import (
    EconomicEvent, MarketRegime, PortfolioState, Position,
    SignalScore, Side, TradeAction, TradeRecommendation,
)

HARD_LIMITS = {
    "max_position_size_pct": 5.0,
    "max_portfolio_heat_pct": 6.0,
    "max_daily_loss_pct": 3.0,
    "max_drawdown_pct": 15.0,
    "max_weekly_loss_pct": 5.0,
    "max_correlated_positions": 3,
    "max_trades_per_day_per_instrument": 10,
    "max_open_positions": 8,
    "min_risk_reward_ratio": 1.5,
    "no_trade_before_news_minutes": 15,
    "no_trade_session_end_minutes": 5,
}
SOFT_LIMITS = {
    "preferred_position_size_pct": 2.0,
    "max_consecutive_losses_alert": 3,
    "max_consecutive_losses_reduce": 5,
    "reduce_size_factor": 0.5,
    "min_minutes_between_trades": 5,
}


def make_portfolio(**kwargs) -> PortfolioState:
    defaults = dict(
        equity=100_000, cash=100_000,
        total_unrealized_pnl=0, total_realized_pnl=0,
        daily_pnl=0, max_drawdown_current=0, portfolio_heat_pct=0,
    )
    defaults.update(kwargs)
    return PortfolioState(**defaults)


def make_signal(instrument: str = "AAPL", rr: float = 2.0) -> SignalScore:
    rec = TradeRecommendation(
        action=TradeAction.BUY, instrument=instrument, confidence=70,
        entry_price=150.0, stop_loss=145.0, take_profit_1=160.0,
        risk_reward_ratio=rr,
    )
    return SignalScore(
        instrument=instrument, timestamp=datetime.utcnow(),
        action=TradeAction.BUY, recommendation=rec,
    )


def test_basic_approval():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    portfolio = make_portfolio()
    result = rm.check_signal(signal, portfolio, [])
    assert result.approved


def test_daily_loss_limit():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    # -3% daily loss = at limit
    portfolio = make_portfolio(daily_pnl=-3000.0)
    result = rm.check_signal(signal, portfolio, [])
    assert not result.approved
    assert "Daily loss" in result.reason


def test_drawdown_kill_switch():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    portfolio = make_portfolio(max_drawdown_current=16.0)
    result = rm.check_signal(signal, portfolio, [])
    assert not result.approved
    assert "drawdown" in result.reason.lower()


def test_max_open_positions():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    positions = [
        Position("MSFT", Side.BUY, 10, 300, 300)
        for _ in range(8)
    ]
    portfolio = make_portfolio(positions=positions)
    result = rm.check_signal(signal, portfolio, [])
    assert not result.approved
    assert "positions" in result.reason.lower()


def test_news_embargo():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    portfolio = make_portfolio()
    # High-impact event in 5 minutes
    from datetime import timedelta
    event = EconomicEvent(
        timestamp=datetime.utcnow() + timedelta(minutes=5),
        name="Fed Rate Decision", country="US", impact="HIGH",
    )
    result = rm.check_signal(signal, portfolio, [event])
    assert not result.approved
    assert "event" in result.reason.lower()


def test_rr_below_minimum():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal(rr=1.0)  # below min 1.5
    portfolio = make_portfolio()
    result = rm.check_signal(signal, portfolio, [])
    assert not result.approved
    assert "R:R" in result.reason


def test_portfolio_heat_limit():
    rm = RiskManager(HARD_LIMITS, SOFT_LIMITS)
    signal = make_signal()
    portfolio = make_portfolio(portfolio_heat_pct=7.0)
    result = rm.check_signal(signal, portfolio, [])
    assert not result.approved
    assert "heat" in result.reason.lower()
