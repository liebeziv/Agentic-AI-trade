"""Momentum strategy agent — trades trend-following signals in trending markets."""
from __future__ import annotations

import polars as pl

from src.strategy.strategies.base_agent import BaseStrategyAgent
from src.strategy.signal_aggregator import SignalAggregator, compute_technical_score
from src.types import (
    MarketRegime, NewsItem, SignalScore, TechnicalSnapshot, TradeAction,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_PARAMS = {
    "rsi_oversold": 35,
    "rsi_overbought": 65,
    "adx_min": 20,           # Only trade when ADX > this (confirming trend)
    "macd_threshold": 0.0,
    "min_composite": 0.30,   # Higher bar than default 0.25 for momentum
}


class MomentumAgent(BaseStrategyAgent):
    """Trend-following specialist. Best in STRONG_TREND regimes."""

    def __init__(self, agent_id: str = "momentum", **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self._params = dict(_DEFAULT_PARAMS)
        self._agg = SignalAggregator()

    @property
    def name(self) -> str:
        return "Momentum"

    @property
    def description(self) -> str:
        return "Trend-following: RSI + MACD + MA alignment + ADX filter"

    @property
    def preferred_regimes(self) -> list[MarketRegime]:
        return [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN,
                MarketRegime.BREAKOUT]

    @property
    def target_instruments(self) -> list[str]:
        return self.config.get("instruments", {}).get("us_stocks", []) + \
               self.config.get("instruments", {}).get("futures", [])

    def get_parameters(self) -> dict:
        return dict(self._params)

    def set_parameters(self, params: dict) -> None:
        self._params.update(params)

    async def generate_signals(
        self,
        market_data: dict[str, pl.DataFrame],
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
    ) -> list[SignalScore]:
        """Generate momentum signals — only fire in trending regimes."""
        if regime not in self.preferred_regimes:
            return []

        signals: list[SignalScore] = []

        for instrument, snap in technical.items():
            ind = snap.indicators
            adx = ind.get("adx_14", 0)
            rsi = ind.get("rsi_14", 50)
            macd_hist = ind.get("macd_hist", 0)
            sma20 = ind.get("sma_20", 0)
            sma50 = ind.get("sma_50", 0)
            close = ind.get("close", 0) or 1

            # ADX filter: only trade strong trends
            if adx < self._params["adx_min"]:
                continue

            # Score: each factor contributes
            score = 0.0
            if rsi < self._params["rsi_oversold"]:
                score += 0.4
            elif rsi > self._params["rsi_overbought"]:
                score -= 0.4

            if macd_hist > self._params["macd_threshold"]:
                score += 0.3
            elif macd_hist < -self._params["macd_threshold"]:
                score -= 0.3

            if sma20 > sma50:
                score += 0.3
            elif sma20 < sma50:
                score -= 0.3

            if abs(score) < self._params["min_composite"]:
                continue

            action = TradeAction.BUY if score > 0 else TradeAction.SELL
            if abs(score) >= 0.7:
                action = TradeAction.STRONG_BUY if score > 0 else TradeAction.STRONG_SELL

            tech_score = compute_technical_score(snap)
            from src.strategy.signal_aggregator import compute_regime_score, compute_composite
            regime_score = compute_regime_score(regime)
            composite, action = compute_composite(
                tech_score, score * 0.5, regime_score, 0.7
            )

            from datetime import datetime
            sig = SignalScore(
                instrument=instrument,
                timestamp=datetime.utcnow(),
                technical_score=tech_score,
                sentiment_score=score * 0.5,
                regime_score=regime_score,
                claude_confidence=0.7,
                composite_score=composite,
                action=action,
            )
            signals.append(sig)
            log.debug("Momentum signal", instrument=instrument,
                      score=round(score, 2), action=action.value, adx=round(adx))

        return signals
