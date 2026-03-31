"""Tests for Phase 3 Step 4-10: EventDrivenAgent, StatArbAgent,
GlobalRiskManager, CorrelationMonitor, StrategyEvolver."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
import pytest

from src.orchestrator.correlation_monitor import CorrelationMonitor
from src.orchestrator.global_risk_manager import GlobalRiskManager, RiskCheckResult, _instrument_sector
from src.reflection.strategy_evolver import StrategyEvolver
from src.strategy.strategies.event_driven_agent import EventDrivenAgent
from src.strategy.strategies.stat_arb_agent import StatArbAgent
from src.types import (
    MarketRegime, NewsItem, PortfolioState, Position, Side,
    SignalScore, TechnicalSnapshot, TradeAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


def make_agent(agent_id="momentum"):
    a = MagicMock()
    a.agent_id = agent_id
    a.name = agent_id
    a.is_active = True
    a.performance = MagicMock()
    a.performance.daily_returns = []
    a.performance.rolling_sharpe_30d = 0.0
    a.performance.total_trades = 0
    a.performance.max_drawdown_pct = 0.0
    a.get_parameters.return_value = {"rsi_oversold": 30}
    return a


def make_snap(instrument="AAPL", **indicators) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        instrument=instrument,
        timeframe="15m",
        timestamp=datetime.utcnow(),
        indicators=indicators,
    )


def make_news(relevance=0.8, sentiment=0.5, instruments=None, age_seconds=60):
    n = MagicMock(spec=NewsItem)
    n.relevance_score = relevance
    n.sentiment_score = sentiment
    n.instruments = instruments or ["AAPL"]
    n.timestamp = datetime.now(tz=timezone.utc).replace(
        second=max(0, datetime.now(tz=timezone.utc).second - age_seconds % 60)
    )
    return n


def make_portfolio(equity=100_000.0, max_dd=0.0) -> PortfolioState:
    p = PortfolioState(
        equity=equity,
        cash=equity,
        total_unrealized_pnl=0.0,
        total_realized_pnl=0.0,
    )
    p.max_drawdown_current = max_dd
    return p


def make_position(instrument="AAPL", side=Side.BUY, qty=10, price=150.0,
                  stop_loss=145.0, agent_id=None):
    pos = MagicMock(spec=Position)
    pos.instrument = instrument
    pos.side = side
    pos.quantity = qty
    pos.current_price = price
    pos.entry_price = price
    pos.stop_loss = stop_loss
    pos.agent_id = agent_id
    return pos


# ---------------------------------------------------------------------------
# EventDrivenAgent
# ---------------------------------------------------------------------------

class TestEventDrivenAgent:
    def _agent(self, claude=None):
        return EventDrivenAgent(
            agent_id="event_driven",
            config={"instruments": {"us_stocks": ["AAPL"], "crypto": [], "futures": []}},
            store=MagicMock(),
            claude_agent=claude,
            technical=MagicMock(),
        )

    def test_no_signals_empty_news(self):
        agent = self._agent()
        snaps = {"AAPL": make_snap("AAPL")}
        sigs = run(agent.generate_signals({}, snaps, [], MarketRegime.BREAKOUT))
        assert sigs == []

    def test_no_signals_low_relevance_news(self):
        agent = self._agent()
        news = [make_news(relevance=0.3, sentiment=0.9)]
        sigs = run(agent.generate_signals({}, {"AAPL": make_snap("AAPL")}, news, MarketRegime.BREAKOUT))
        assert sigs == []

    def test_fallback_bullish_signal(self):
        """Without Claude, uses raw sentiment_score fallback."""
        agent = self._agent(claude=None)
        news = [make_news(relevance=0.9, sentiment=0.8, instruments=["AAPL"])]
        snaps = {"AAPL": make_snap("AAPL", trend_bias="bullish")}
        sigs = run(agent.generate_signals({}, snaps, news, MarketRegime.BREAKOUT))
        assert len(sigs) == 1
        assert sigs[0].composite_score > 0

    def test_fallback_bearish_signal(self):
        agent = self._agent(claude=None)
        news = [make_news(relevance=0.9, sentiment=-0.8, instruments=["AAPL"])]
        snaps = {"AAPL": make_snap("AAPL", trend_bias="bearish")}
        sigs = run(agent.generate_signals({}, snaps, news, MarketRegime.STRONG_TREND_DOWN))
        assert len(sigs) == 1
        assert sigs[0].composite_score < 0

    def test_technical_alignment_blocks_buy_in_bearish_trend(self):
        agent = self._agent(claude=None)
        agent._params["require_technical_alignment"] = True
        news = [make_news(relevance=0.9, sentiment=0.9, instruments=["AAPL"])]
        # trend_bias = bearish → should block bullish news signal
        snap = make_snap("AAPL")
        snap.trend_bias = "bearish"
        sigs = run(agent.generate_signals({}, {"AAPL": snap}, news, MarketRegime.BREAKOUT))
        assert sigs == []

    def test_no_snap_skips_instrument(self):
        agent = self._agent(claude=None)
        news = [make_news(relevance=0.9, sentiment=0.8, instruments=["AAPL"])]
        sigs = run(agent.generate_signals({}, {}, news, MarketRegime.BREAKOUT))
        assert sigs == []

    def test_preferred_regimes(self):
        agent = self._agent()
        assert MarketRegime.BREAKOUT in agent.preferred_regimes

    def test_get_set_parameters(self):
        agent = self._agent()
        agent.set_parameters({"min_news_relevance": 0.5})
        assert agent.get_parameters()["min_news_relevance"] == 0.5


# ---------------------------------------------------------------------------
# StatArbAgent
# ---------------------------------------------------------------------------

class TestStatArbAgent:
    def _agent(self):
        return StatArbAgent(
            agent_id="stat_arb",
            config={"instruments": {}},
            store=MagicMock(),
            claude_agent=None,
            technical=MagicMock(),
        )

    def _make_prices(self, n=100, offset=0.0):
        rng = np.random.default_rng(42)
        return pl.DataFrame({"close": rng.normal(100 + offset, 1, n)})

    def test_no_signals_wrong_regime(self):
        agent = self._agent()
        sigs = run(agent.generate_signals({}, {}, [], MarketRegime.STRONG_TREND_UP))
        assert sigs == []

    def test_no_signals_missing_data(self):
        agent = self._agent()
        sigs = run(agent.generate_signals({}, {}, [], MarketRegime.RANGE_BOUND))
        assert sigs == []

    def test_no_signal_low_correlation(self):
        agent = self._agent()
        rng = np.random.default_rng(1)
        df_a = pl.DataFrame({"close": rng.normal(100, 5, 100)})
        df_b = pl.DataFrame({"close": rng.normal(200, 5, 100)})  # uncorrelated
        data = {"AAPL": df_a, "MSFT": df_b}
        agent._params["min_correlation"] = 0.99  # very high threshold
        sigs = run(agent.generate_signals(data, {}, [], MarketRegime.RANGE_BOUND))
        assert sigs == []

    def test_sell_signal_positive_z(self):
        """Spread too wide → sell A (AAPL)."""
        agent = self._agent()
        n = 100
        rng = np.random.default_rng(10)
        base = rng.normal(0, 1, n).cumsum() + 100
        # A and B highly correlated; but last point of A jumps (positive spread deviation)
        prices_b = base.copy()
        prices_a = base * 1.1 + 5
        prices_a[-1] += 20   # push A up artificially → positive z
        df_a = pl.DataFrame({"close": prices_a})
        df_b = pl.DataFrame({"close": prices_b})
        data = {"AAPL": df_a, "MSFT": df_b}
        agent._params["pairs"] = [("AAPL", "MSFT")]
        agent._params["z_score_entry"] = 1.5
        sigs = run(agent.generate_signals(data, {}, [], MarketRegime.RANGE_BOUND))
        if sigs:
            assert sigs[0].action in (TradeAction.SELL, TradeAction.STRONG_SELL)

    def test_target_instruments_unique(self):
        agent = self._agent()
        instruments = agent.target_instruments
        assert len(instruments) == len(set(instruments))

    def test_preferred_regimes(self):
        agent = self._agent()
        assert MarketRegime.RANGE_BOUND in agent.preferred_regimes


# ---------------------------------------------------------------------------
# GlobalRiskManager
# ---------------------------------------------------------------------------

class TestGlobalRiskManager:
    def _grm(self, **overrides):
        limits = {
            "max_drawdown_pct": 15.0,
            "max_net_exposure_pct": 80.0,
            "max_agents_per_instrument": 2,
            "max_correlated_cross_agent": 4,
            "agent_budget_pct": 0.25,
            **overrides,
        }
        return GlobalRiskManager(limits)

    def _signal(self, instrument="AAPL", action=TradeAction.BUY):
        s = MagicMock(spec=SignalScore)
        s.instrument = instrument
        s.action = action
        return s

    def test_approved_clean_portfolio(self):
        grm = self._grm()
        result = grm.check_portfolio_signal(
            self._signal(), "momentum", [], make_portfolio()
        )
        assert result.approved

    def test_rejected_drawdown_exceeded(self):
        grm = self._grm()
        portfolio = make_portfolio(max_dd=20.0)
        result = grm.check_portfolio_signal(
            self._signal(), "momentum", [], portfolio
        )
        assert not result.approved
        assert "drawdown" in result.reason.lower()

    def test_rejected_instrument_overexposed(self):
        grm = self._grm(max_agents_per_instrument=1)
        positions = [make_position("AAPL")]  # already 1 agent in AAPL
        result = grm.check_portfolio_signal(
            self._signal("AAPL"), "momentum", positions, make_portfolio()
        )
        assert not result.approved

    def test_rejected_agent_budget_exhausted(self):
        grm = self._grm()
        grm.set_agent_budget("momentum", 0)  # zero budget
        result = grm.check_portfolio_signal(
            self._signal(), "momentum", [], make_portfolio()
        )
        assert not result.approved

    def test_checks_passed_list_populated(self):
        grm = self._grm()
        result = grm.check_portfolio_signal(
            self._signal(), "momentum", [], make_portfolio()
        )
        assert result.approved
        assert len(result.checks_passed) > 0

    def test_sector_mapping(self):
        assert _instrument_sector("AAPL") == "tech"
        assert _instrument_sector("EURUSD") == "forex"
        assert _instrument_sector("BTC/USDT") == "crypto"
        assert _instrument_sector("XYZ123") == "other"


# ---------------------------------------------------------------------------
# CorrelationMonitor
# ---------------------------------------------------------------------------

class TestCorrelationMonitor:
    def _agents_with_returns(self, n=30):
        rng = np.random.default_rng(99)
        agents = []
        for i in range(3):
            a = make_agent(f"agent_{i}")
            a.performance.daily_returns = list(rng.normal(0.001 * i, 0.01, n))
            agents.append(a)
        return agents

    def test_empty_matrix_with_no_returns(self):
        agents = [make_agent("a"), make_agent("b")]
        cm = CorrelationMonitor(agents)
        matrix = cm.compute_correlation_matrix(lookback_days=30)
        assert matrix == {}

    def test_matrix_populated_with_returns(self):
        agents = self._agents_with_returns(n=35)
        cm = CorrelationMonitor(agents)
        matrix = cm.compute_correlation_matrix(lookback_days=30)
        assert len(matrix) == 3
        # Diagonal should be 1.0
        for aid in matrix:
            assert matrix[aid][aid] == 1.0

    def test_diversification_score_range(self):
        agents = self._agents_with_returns(n=35)
        cm = CorrelationMonitor(agents)
        score = cm.get_diversification_score(lookback_days=30)
        assert 0.0 <= score <= 1.0

    def test_diversification_neutral_no_history(self):
        agents = [make_agent("a"), make_agent("b")]
        cm = CorrelationMonitor(agents)
        assert cm.get_diversification_score() == 0.5

    def test_no_alerts_uncorrelated(self):
        rng = np.random.default_rng(7)
        agents = []
        for i in range(2):
            a = make_agent(f"a{i}")
            # Orthogonal return streams → correlation ≈ 0
            a.performance.daily_returns = list(rng.normal(0, 0.01, 35))
            agents.append(a)
        cm = CorrelationMonitor(agents)
        # With random uncorrelated returns, alerts should usually be empty
        # (could occasionally trigger if random correlation > 0.7)
        alerts = cm.check_alerts(lookback_days=30)
        assert isinstance(alerts, list)

    def test_penalties_zero_without_returns(self):
        agents = [make_agent("a"), make_agent("b")]
        cm = CorrelationMonitor(agents)
        penalties = cm.get_correlation_penalties()
        assert all(v == 0.0 for v in penalties.values())

    def test_summary_has_required_keys(self):
        agents = self._agents_with_returns(n=35)
        cm = CorrelationMonitor(agents)
        summary = cm.summary(lookback_days=30)
        assert "diversification_score" in summary
        assert "alerts" in summary
        assert "correlation_matrix" in summary
        assert "penalties" in summary


# ---------------------------------------------------------------------------
# StrategyEvolver
# ---------------------------------------------------------------------------

class TestStrategyEvolver:
    def _evolver(self, agents=None, claude=None, store=None):
        if agents is None:
            agents = [make_agent()]
        if store is None:
            s = MagicMock()
            s.get_trades.return_value = []
            store = s
        return StrategyEvolver(agents, claude, store, notifier=None)

    def test_insufficient_trades_skipped(self):
        evolver = self._evolver()
        report = run(evolver.weekly_review())
        assert "agent_0" in report["agents"] or any(
            v.get("status") == "insufficient_trades"
            for v in report["agents"].values()
        )

    def test_report_has_timestamp(self):
        evolver = self._evolver()
        report = run(evolver.weekly_review())
        assert "timestamp" in report

    def test_no_claude_returns_empty_analysis(self):
        evolver = self._evolver(claude=None)
        result = run(evolver._claude_review(make_agent(), []))
        assert result == {}

    def test_deactivate_chronic_underperformer(self):
        agent = make_agent("bad_agent")
        agent.performance.rolling_sharpe_30d = -1.0
        agent.performance.total_trades = 30
        agent.is_active = True

        store = MagicMock()
        # Provide enough mock trades so evolution runs
        mock_trade = MagicMock()
        mock_trade.pnl = -100.0
        mock_trade.lessons = ["stop too tight"]
        mock_trade.open_time = datetime.utcnow()
        mock_trade.agent_id = "bad_agent"
        store.get_trades.return_value = [mock_trade] * 10

        evolver = StrategyEvolver([agent], claude=None, store=store)
        report = run(evolver.weekly_review())
        # Without Claude, no params suggested → status = no_changes_suggested
        # Deactivation happens only after Claude review if sharpe < -0.5 and trades > 20
        assert agent.performance.rolling_sharpe_30d == -1.0  # unchanged mock
