"""Position sizer — Kelly, fixed fractional, ATR-based."""
from __future__ import annotations

import numpy as np

from src.types import PortfolioState, PositionSize, SignalScore
from src.utils.logger import get_logger

log = get_logger(__name__)


class PositionSizer:
    def __init__(self, sizing_config: dict, hard_limits: dict, store=None) -> None:
        self.config = sizing_config
        self.limits = hard_limits
        self.store = store

    def calculate(
        self,
        signal: SignalScore,
        portfolio: PortfolioState,
        instrument_config: dict,
        current_atr: float = 0.0,
    ) -> PositionSize:
        method = self.config.get("method", "half_kelly")
        equity = max(portfolio.equity, 1.0)
        rec = signal.recommendation

        if method == "half_kelly":
            risk_pct = self._kelly_fraction(equity)
        elif method == "atr_based" and current_atr > 0 and rec:
            stop_distance = current_atr * self.config.get("atr_risk_multiple", 2.0)
            risk_amount = equity * (self.config.get("fixed_fraction_pct", 2.0) / 100)
            units = risk_amount / stop_distance if stop_distance > 0 else 0
            risk_pct = (units * (rec.entry_price or 1)) / equity
        else:
            risk_pct = self.config.get("fixed_fraction_pct", 2.0) / 100

        # Apply hard cap
        max_pct = self.limits.get("max_position_size_pct", 5.0) / 100
        risk_pct = min(risk_pct, max_pct)
        risk_pct = max(risk_pct, 0.005)  # never below 0.5%

        risk_amount = equity * risk_pct

        # Convert risk amount to units
        if rec and rec.stop_loss and rec.entry_price:
            stop_dist = abs(rec.entry_price - rec.stop_loss)
            units = risk_amount / stop_dist if stop_dist > 0 else (risk_amount / rec.entry_price)
            notional = units * rec.entry_price
        else:
            entry = (rec.entry_price if rec and rec.entry_price else 1.0)
            units = risk_amount / entry
            notional = risk_amount

        # Apply USD limits
        min_usd = self.limits.get("min_position_size_usd", 100)
        max_usd = self.limits.get("max_position_size_usd", 50000)
        if notional < min_usd:
            scale = min_usd / notional
            units *= scale
            notional = min_usd
        elif notional > max_usd:
            scale = max_usd / notional
            units *= scale
            notional = max_usd

        log.debug(
            "Position size calculated",
            instrument=signal.instrument,
            method=method,
            risk_pct=round(risk_pct * 100, 2),
            units=round(units, 4),
            notional=round(notional, 2),
        )

        return PositionSize(
            instrument=signal.instrument,
            units=round(units, 4),
            notional_value=round(notional, 2),
            risk_amount=round(risk_amount, 2),
            risk_pct_of_equity=round(risk_pct * 100, 2),
            method_used=method,
            kelly_fraction=risk_pct if method == "half_kelly" else None,
        )

    def _kelly_fraction(self, equity: float) -> float:
        """Compute half-Kelly fraction from recent trade history."""
        default_pct = self.config.get("fixed_fraction_pct", 2.0) / 100

        if not self.store:
            return default_pct

        lookback = self.config.get("kelly_lookback_trades", 50)
        trades = self.store.get_trades(limit=lookback)

        if len(trades) < 10:
            return default_pct

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        if not wins or not losses:
            return default_pct

        win_rate = len(wins) / len(trades)
        avg_win = float(np.mean([t.pnl_pct for t in wins]))
        avg_loss = abs(float(np.mean([t.pnl_pct for t in losses])))

        if avg_loss == 0:
            return default_pct

        b = avg_win / avg_loss
        kelly = (b * win_rate - (1 - win_rate)) / b

        # Half-Kelly and clamp
        half_kelly = kelly * 0.5
        min_pct = 0.005
        max_pct = self.limits.get("max_position_size_pct", 5.0) / 100
        return max(min_pct, min(half_kelly, max_pct))
