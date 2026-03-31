"""Mean-reversion strategy agent — fades extremes in range-bound markets."""
from __future__ import annotations

import polars as pl

from src.strategy.strategies.base_agent import BaseStrategyAgent, _extract_symbols
from src.strategy.signal_aggregator import compute_technical_score, compute_composite, compute_regime_score
from src.types import (
    MarketRegime, NewsItem, SignalScore, TechnicalSnapshot, TradeAction,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_PARAMS = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bb_lower_pct": 0.05,    # price within 5% of lower band = buy signal
    "bb_upper_pct": 0.05,    # price within 5% of upper band = sell signal
    "adx_max": 22,           # Only trade when ADX < this (not trending)
    "stoch_oversold": 20,
    "stoch_overbought": 80,
    "min_score": 0.25,
}


class MeanReversionAgent(BaseStrategyAgent):
    """Mean-reversion specialist. Best in RANGE_BOUND and low-ADX markets."""

    def __init__(self, agent_id: str = "mean_reversion", **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self._params = dict(_DEFAULT_PARAMS)

    @property
    def name(self) -> str:
        return "Mean Reversion"

    @property
    def description(self) -> str:
        return "Fade extremes: RSI + Bollinger Band touch + Stochastic in range-bound markets"

    @property
    def preferred_regimes(self) -> list[MarketRegime]:
        return [MarketRegime.RANGE_BOUND, MarketRegime.WEAK_TREND]

    @property
    def target_instruments(self) -> list[str]:
        cfg = self.config.get("instruments", {})
        return _extract_symbols(cfg.get("forex", []) + cfg.get("us_stocks", []))

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
        """Generate mean-reversion signals — only in non-trending markets."""
        if regime not in self.preferred_regimes:
            return []

        signals: list[SignalScore] = []

        for instrument, snap in technical.items():
            ind = snap.indicators
            adx = ind.get("adx_14", 25)
            rsi = ind.get("rsi_14", 50)
            stoch_k = ind.get("stoch_k", 50)
            close = ind.get("close", 0) or 1
            bb_upper = ind.get("bbands_upper", close * 1.02)
            bb_lower = ind.get("bbands_lower", close * 0.98)
            bb_mid = ind.get("bbands_mid", close)

            # ADX filter: only trade when market is NOT trending
            if adx > self._params["adx_max"]:
                continue

            score = 0.0

            # RSI extremes
            if rsi < self._params["rsi_oversold"]:
                score += 0.4
            elif rsi > self._params["rsi_overbought"]:
                score -= 0.4
            else:
                continue  # No extreme — skip

            # Bollinger Band confirmation
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                if close < bb_lower + bb_range * self._params["bb_lower_pct"]:
                    score += 0.3   # near lower band → buy
                elif close > bb_upper - bb_range * self._params["bb_upper_pct"]:
                    score -= 0.3   # near upper band → sell

            # Stochastic confirmation
            if stoch_k < self._params["stoch_oversold"]:
                score += 0.2
            elif stoch_k > self._params["stoch_overbought"]:
                score -= 0.2

            if abs(score) < self._params["min_score"]:
                continue

            tech_score = compute_technical_score(snap)
            regime_score = compute_regime_score(regime)
            # For mean reversion, sentiment is contrarian to momentum
            sentiment = score * 0.6
            composite, action = compute_composite(tech_score, sentiment, regime_score, 0.65)

            from datetime import datetime
            sig = SignalScore(
                instrument=instrument,
                timestamp=datetime.utcnow(),
                technical_score=tech_score,
                sentiment_score=sentiment,
                regime_score=regime_score,
                claude_confidence=0.65,
                composite_score=composite,
                action=action,
            )
            signals.append(sig)
            log.debug("MeanReversion signal", instrument=instrument,
                      score=round(score, 2), action=action.value, rsi=round(rsi))

        return signals
