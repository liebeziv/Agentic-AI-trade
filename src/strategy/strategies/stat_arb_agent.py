"""Statistical Arbitrage agent — trades mean-reverting spreads between correlated pairs."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from src.strategy.strategies.base_agent import BaseStrategyAgent
from src.types import (
    MarketRegime, NewsItem, SignalScore, TechnicalSnapshot, TradeAction,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_PARAMS = {
    "pairs": [
        ("AAPL", "MSFT"),
        ("GOOGL", "META"),
        ("SPY", "QQQ"),
        ("BTC/USDT", "ETH/USDT"),
    ],
    "lookback_period": 60,        # bars for correlation / hedge ratio
    "z_score_entry": 2.0,
    "z_score_exit": 0.5,
    "z_score_stop": 3.5,
    "min_correlation": 0.70,
    "min_score": 0.30,
}


class StatArbAgent(BaseStrategyAgent):
    """Statistical arbitrage specialist — range-bound / low-volatility pairs trading."""

    def __init__(self, agent_id: str = "stat_arb", **kwargs) -> None:
        super().__init__(agent_id, **kwargs)
        self._params = dict(_DEFAULT_PARAMS)
        # Per-pair hedge ratios (updated each cycle)
        self._hedge_ratios: dict[tuple, float] = {}

    @property
    def name(self) -> str:
        return "Statistical Arbitrage"

    @property
    def description(self) -> str:
        return "Trades correlated pairs when spread Z-score exceeds ±2σ"

    @property
    def preferred_regimes(self) -> list[MarketRegime]:
        return [MarketRegime.RANGE_BOUND, MarketRegime.WEAK_TREND]

    @property
    def target_instruments(self) -> list[str]:
        instruments: list[str] = []
        for pair in self._params["pairs"]:
            instruments.extend(pair)
        return list(dict.fromkeys(instruments))  # deduplicated

    def get_parameters(self) -> dict:
        return dict(self._params)

    def set_parameters(self, params: dict) -> None:
        self._params.update(params)

    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        market_data: dict[str, pl.DataFrame],
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
    ) -> list[SignalScore]:
        """Generate pair-spread signals."""
        if regime not in self.preferred_regimes:
            return []

        signals: list[SignalScore] = []
        lookback = self._params["lookback_period"]

        for pair in self._params["pairs"]:
            asset_a, asset_b = pair
            df_a = market_data.get(asset_a)
            df_b = market_data.get(asset_b)
            if df_a is None or df_b is None:
                continue
            if df_a.is_empty() or df_b.is_empty():
                continue

            prices_a = df_a["close"].to_numpy()
            prices_b = df_b["close"].to_numpy()

            # Align lengths
            min_len = min(len(prices_a), len(prices_b))
            if min_len < lookback + 5:
                continue
            prices_a = prices_a[-min_len:]
            prices_b = prices_b[-min_len:]

            # Correlation check
            corr_window = min(lookback, min_len)
            corr = float(np.corrcoef(
                prices_a[-corr_window:], prices_b[-corr_window:]
            )[0, 1])
            if abs(corr) < self._params["min_correlation"]:
                continue

            # OLS hedge ratio: prices_a = beta * prices_b + alpha
            pb = prices_b[-lookback:]
            pa = prices_a[-lookback:]
            beta = float(np.polyfit(pb, pa, 1)[0])
            alpha = float(np.mean(pa) - beta * np.mean(pb))
            self._hedge_ratios[pair] = beta

            # Spread & Z-score
            spread = prices_a - beta * prices_b - alpha
            spread_win = spread[-lookback:]
            spread_mean = float(np.mean(spread_win))
            spread_std = float(np.std(spread_win))
            if spread_std < 1e-10:
                continue
            z = float((spread[-1] - spread_mean) / spread_std)

            log.debug("StatArb spread", pair=f"{asset_a}/{asset_b}",
                      z=round(z, 2), corr=round(corr, 2), beta=round(beta, 3))

            # Entry logic
            entry = self._params["z_score_entry"]
            stop = self._params["z_score_stop"]
            min_score = self._params["min_score"]

            if abs(z) < entry or abs(z) > stop:
                continue  # below entry or past hard stop

            # Normalised composite: 1.0 at z_score_stop
            magnitude = min(abs(z) / stop, 1.0)
            if magnitude < min_score:
                continue

            if z > entry:
                # Spread too wide: short A (overbought relative to B) → SELL A
                action = TradeAction.SELL if magnitude < 0.7 else TradeAction.STRONG_SELL
                composite = -magnitude
            else:
                # Spread too narrow: long A (oversold relative to B) → BUY A
                action = TradeAction.BUY if magnitude < 0.7 else TradeAction.STRONG_BUY
                composite = magnitude

            sig = SignalScore(
                instrument=asset_a,
                timestamp=datetime.utcnow(),
                technical_score=composite,
                sentiment_score=0.0,
                regime_score=0.2,   # low — pure stat signal
                claude_confidence=0.60,
                composite_score=composite,
                action=action,
            )
            signals.append(sig)
            log.info("StatArb signal",
                     pair=f"{asset_a}/{asset_b}", z=round(z, 2),
                     action=action.value, composite=round(composite, 3))

        return signals
