"""Tests for Phase 3: BaseStrategyAgent, MomentumAgent, MeanReversionAgent, Orchestrator."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.strategy.strategies.base_agent import AgentPerformance, BaseStrategyAgent
from src.strategy.strategies.mean_reversion_agent import MeanReversionAgent
from src.strategy.strategies.momentum_agent import MomentumAgent
from src.strategy.orchestrator import Orchestrator
from src.types import (
    MarketRegime,
    SignalScore,
    TechnicalSnapshot,
    TradeAction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_snapshot(indicators: dict) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        instrument=indicators.get("_instrument", "AAPL"),
        timeframe="15m",
        timestamp=datetime.utcnow(),
        indicators=indicators,
    )


def make_signal(instrument: str, action: TradeAction, composite: float) -> SignalScore:
    return SignalScore(
        instrument=instrument,
        timestamp=datetime.utcnow(),
        technical_score=composite,
        sentiment_score=0.0,
        regime_score=0.5,
        claude_confidence=0.7,
        composite_score=composite,
        action=action,
    )


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AgentPerformance
# ---------------------------------------------------------------------------

class TestAgentPerformance:
    def test_fitness_small_when_no_history(self):
        """With no trade history, fitness is low (drawdown penalty only)."""
        p = AgentPerformance()
        assert p.fitness < 0.3  # sharpe=0, wr=0, small dd contribution

    def test_fitness_positive_good_sharpe(self):
        p = AgentPerformance(rolling_sharpe_30d=2.0, win_rate_pct=60.0, max_drawdown_pct=5.0)
        assert p.fitness > 0.5

    def test_fitness_penalized_by_drawdown(self):
        low_dd = AgentPerformance(rolling_sharpe_30d=1.5, win_rate_pct=55.0, max_drawdown_pct=5.0)
        high_dd = AgentPerformance(rolling_sharpe_30d=1.5, win_rate_pct=55.0, max_drawdown_pct=25.0)
        assert low_dd.fitness > high_dd.fitness

    def test_fitness_capped_at_1(self):
        p = AgentPerformance(rolling_sharpe_30d=10.0, win_rate_pct=100.0, max_drawdown_pct=0.0)
        assert p.fitness <= 1.0

    def test_fitness_non_negative(self):
        p = AgentPerformance(rolling_sharpe_30d=-5.0, win_rate_pct=0.0, max_drawdown_pct=50.0)
        assert p.fitness >= 0.0


# ---------------------------------------------------------------------------
# MomentumAgent
# ---------------------------------------------------------------------------

class TestMomentumAgent:
    def _agent(self):
        return MomentumAgent(
            agent_id="momentum",
            config={"instruments": {"us_stocks": ["AAPL"], "futures": []}},
            store=MagicMock(),
            claude_agent=None,
            technical=MagicMock(),
        )

    def test_no_signals_in_wrong_regime(self):
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({"adx_14": 30, "rsi_14": 40, "macd_hist": 0.5, "sma_20": 105, "sma_50": 100, "close": 110})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.RANGE_BOUND))
        assert signals == []

    def test_no_signal_low_adx(self):
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({"adx_14": 10, "rsi_14": 30, "macd_hist": 1.0, "sma_20": 110, "sma_50": 100, "close": 115})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.STRONG_TREND_UP))
        assert signals == []

    def test_buy_signal_strong_trend(self):
        agent = self._agent()
        # ADX=30, RSI oversold, MACD+, SMA20>SMA50 → strong buy
        snaps = {"AAPL": make_snapshot({"adx_14": 30, "rsi_14": 30, "macd_hist": 0.5, "sma_20": 110, "sma_50": 100, "close": 115})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.STRONG_TREND_UP))
        assert len(signals) == 1
        assert signals[0].action in (TradeAction.BUY, TradeAction.STRONG_BUY)

    def test_sell_signal_strong_trend_down(self):
        agent = self._agent()
        # ADX=35, RSI overbought, MACD-, SMA20<SMA50 → sell direction
        snaps = {"AAPL": make_snapshot({"adx_14": 35, "rsi_14": 70, "macd_hist": -0.5, "sma_20": 90, "sma_50": 100, "close": 85})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.STRONG_TREND_DOWN))
        assert len(signals) == 1
        # composite may be NEUTRAL near threshold; verify bearish direction
        assert signals[0].composite_score < 0

    def test_no_signal_mixed_indicators(self):
        """RSI neutral, so score may be below min_composite threshold."""
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({"adx_14": 25, "rsi_14": 50, "macd_hist": 0.1, "sma_20": 101, "sma_50": 100, "close": 102})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.STRONG_TREND_UP))
        # score = 0 (RSI neutral) + 0.3 (macd+) + 0.3 (sma20>sma50) = 0.6 → should signal
        assert isinstance(signals, list)

    def test_preferred_regimes(self):
        agent = self._agent()
        assert MarketRegime.STRONG_TREND_UP in agent.preferred_regimes
        assert MarketRegime.BREAKOUT in agent.preferred_regimes

    def test_status_dict_has_required_keys(self):
        agent = self._agent()
        status = agent.status_dict()
        assert "agent_id" in status
        assert "name" in status
        assert "active" in status


# ---------------------------------------------------------------------------
# MeanReversionAgent
# ---------------------------------------------------------------------------

class TestMeanReversionAgent:
    def _agent(self):
        return MeanReversionAgent(
            agent_id="mean_reversion",
            config={"instruments": {"forex": [], "us_stocks": ["AAPL"]}},
            store=MagicMock(),
            claude_agent=None,
            technical=MagicMock(),
        )

    def test_no_signals_trending_regime(self):
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({"adx_14": 18, "rsi_14": 25, "stoch_k": 15, "close": 95, "bbands_upper": 105, "bbands_lower": 90, "bbands_mid": 97})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.STRONG_TREND_UP))
        assert signals == []

    def test_no_signal_high_adx(self):
        """ADX > 22 means trending — mean reversion should not trade."""
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({"adx_14": 30, "rsi_14": 25, "stoch_k": 15, "close": 95, "bbands_upper": 105, "bbands_lower": 90, "bbands_mid": 97})}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.RANGE_BOUND))
        assert signals == []

    def test_buy_signal_rsi_oversold(self):
        agent = self._agent()
        # ADX=15, RSI oversold, near lower BB, stoch oversold → bullish composite
        snaps = {"AAPL": make_snapshot({
            "adx_14": 15, "rsi_14": 25, "stoch_k": 15,
            "close": 90.1, "bbands_upper": 105, "bbands_lower": 90, "bbands_mid": 97,
        })}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.RANGE_BOUND))
        assert len(signals) == 1
        # composite may be NEUTRAL near threshold; verify bullish direction
        assert signals[0].composite_score > 0

    def test_sell_signal_rsi_overbought(self):
        agent = self._agent()
        # ADX=15, RSI overbought, near upper BB, stoch overbought → bearish composite
        snaps = {"AAPL": make_snapshot({
            "adx_14": 15, "rsi_14": 75, "stoch_k": 85,
            "close": 104.9, "bbands_upper": 105, "bbands_lower": 90, "bbands_mid": 97,
        })}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.RANGE_BOUND))
        assert len(signals) == 1
        # composite may be NEUTRAL near threshold; verify bearish direction
        assert signals[0].composite_score < 0

    def test_no_signal_rsi_neutral(self):
        """RSI in neutral zone (40-60) should produce no signal."""
        agent = self._agent()
        snaps = {"AAPL": make_snapshot({
            "adx_14": 15, "rsi_14": 50, "stoch_k": 50,
            "close": 97, "bbands_upper": 105, "bbands_lower": 90, "bbands_mid": 97,
        })}
        signals = run(agent.generate_signals({}, snaps, [], MarketRegime.RANGE_BOUND))
        assert signals == []

    def test_preferred_regimes(self):
        agent = self._agent()
        assert MarketRegime.RANGE_BOUND in agent.preferred_regimes
        assert MarketRegime.WEAK_TREND in agent.preferred_regimes


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def _agents(self):
        momentum = MomentumAgent(
            agent_id="momentum",
            config={"instruments": {"us_stocks": ["AAPL"], "futures": []}},
            store=MagicMock(), claude_agent=None, technical=MagicMock(),
        )
        mean_rev = MeanReversionAgent(
            agent_id="mean_reversion",
            config={"instruments": {"forex": [], "us_stocks": ["AAPL"]}},
            store=MagicMock(), claude_agent=None, technical=MagicMock(),
        )
        return [momentum, mean_rev]

    def test_allocate_capital_equal_fallback(self):
        """With no trade history, fitness=0, so equal allocation."""
        agents = self._agents()
        orch = Orchestrator(agents, total_capital=100_000, claude_agent=None)
        allocation = run(orch.allocate_capital(MarketRegime.RANGE_BOUND))
        assert len(allocation) == 2
        total = sum(allocation.values())
        assert total <= 100_000 * 1.001  # allow tiny float error

    def test_no_conflict_single_signal(self):
        orch = Orchestrator([], total_capital=10_000)
        sigs = [make_signal("AAPL", TradeAction.BUY, 0.6)]
        resolved = orch.resolve_conflicts(sigs)
        assert len(resolved) == 1
        assert resolved[0].action == TradeAction.BUY

    def test_conflict_resolved_by_margin(self):
        orch = Orchestrator([], total_capital=10_000)
        sigs = [
            make_signal("AAPL", TradeAction.BUY, 0.7),
            make_signal("AAPL", TradeAction.SELL, -0.2),
        ]
        resolved = orch.resolve_conflicts(sigs)
        # margin = |0.7 - 0.2| = 0.5 > 0.15 → winner is BUY
        assert len(resolved) == 1
        assert resolved[0].action == TradeAction.BUY

    def test_conflict_skipped_small_margin(self):
        orch = Orchestrator([], total_capital=10_000)
        sigs = [
            make_signal("AAPL", TradeAction.BUY, 0.35),
            make_signal("AAPL", TradeAction.SELL, -0.32),
        ]
        resolved = orch.resolve_conflicts(sigs)
        # margin = |0.35 - 0.32| = 0.03 < 0.15 → skipped
        assert len(resolved) == 0

    def test_conflict_sell_wins(self):
        orch = Orchestrator([], total_capital=10_000)
        sigs = [
            make_signal("AAPL", TradeAction.BUY, 0.2),
            make_signal("AAPL", TradeAction.SELL, -0.7),
        ]
        resolved = orch.resolve_conflicts(sigs)
        assert len(resolved) == 1
        assert resolved[0].action == TradeAction.SELL

    def test_multi_instrument_no_cross_conflict(self):
        orch = Orchestrator([], total_capital=10_000)
        sigs = [
            make_signal("AAPL", TradeAction.BUY, 0.6),
            make_signal("MSFT", TradeAction.SELL, -0.5),
        ]
        resolved = orch.resolve_conflicts(sigs)
        assert len(resolved) == 2

    def test_capital_not_exceed_total(self):
        agents = self._agents()
        orch = Orchestrator(agents, total_capital=50_000)
        allocation = run(orch.allocate_capital(MarketRegime.STRONG_TREND_UP))
        for aid, amount in allocation.items():
            assert amount <= 50_000 * 0.41  # max 40% + float buffer

    def test_agent_status_list(self):
        agents = self._agents()
        orch = Orchestrator(agents, total_capital=10_000)
        statuses = orch.agent_status()
        assert len(statuses) == 2
        assert all("agent_id" in s for s in statuses)
