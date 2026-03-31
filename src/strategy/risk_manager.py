"""Risk manager — enforces hard and soft limits before order execution."""
from __future__ import annotations
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.types import (
    EconomicEvent, PortfolioState, RiskCheckResult, SignalScore, TradeRecord,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

CORRELATED_GROUPS = [
    {"EURUSD", "GBPUSD", "AUDUSD"},       # Risk-on / USD pairs
    {"USDJPY"},                             # Safe-haven JPY
    {"GC", "SI"},                          # Precious metals
    {"CL", "NG"},                          # Energy
    {"ES", "NQ", "SPY"},                   # US equity indices
    {"BTC/USDT", "ETH/USDT", "SOL/USDT"}, # Crypto
]


def _find_correlated_group(instrument: str) -> set[str]:
    for group in CORRELATED_GROUPS:
        if instrument in group:
            return group
    return {instrument}


class RiskManager:
    def __init__(self, hard_limits: dict, soft_limits: dict, store=None) -> None:
        self.limits = hard_limits
        self.soft = soft_limits
        self.store = store
        # Track per-instrument trade count today
        self._today_trades: dict[str, int] = {}
        self._last_trade_time: dict[str, datetime] = {}
        self._reset_date: datetime = datetime.utcnow().date()  # type: ignore[assignment]

    def _reset_daily_counters_if_needed(self) -> None:
        today = datetime.utcnow().date()
        if today != self._reset_date:
            self._today_trades = {}
            self._reset_date = today  # type: ignore[assignment]

    def check_signal(
        self,
        signal: SignalScore,
        portfolio: PortfolioState,
        calendar: list[EconomicEvent],
    ) -> RiskCheckResult:
        self._reset_daily_counters_if_needed()

        checks_passed: list[str] = []
        instrument = signal.instrument

        # 1. Daily loss limit
        daily_loss_pct = abs(portfolio.daily_pnl) / max(portfolio.equity, 1) * 100
        if portfolio.daily_pnl < 0 and daily_loss_pct >= self.limits.get("max_daily_loss_pct", 3.0):
            return RiskCheckResult(approved=False, reason=f"Daily loss limit {daily_loss_pct:.1f}% reached")
        checks_passed.append("daily_loss_ok")

        # 2. Max drawdown kill switch
        if portfolio.max_drawdown_current >= self.limits.get("max_drawdown_pct", 15.0):
            return RiskCheckResult(approved=False, reason="Max drawdown kill switch triggered")
        checks_passed.append("drawdown_ok")

        # 3. Portfolio heat
        if portfolio.portfolio_heat_pct >= self.limits.get("max_portfolio_heat_pct", 6.0):
            return RiskCheckResult(approved=False, reason=f"Portfolio heat {portfolio.portfolio_heat_pct:.1f}% at limit")
        checks_passed.append("heat_ok")

        # 4. Max open positions
        if len(portfolio.positions) >= self.limits.get("max_open_positions", 8):
            return RiskCheckResult(approved=False, reason="Max open positions reached")
        checks_passed.append("positions_ok")

        # 5. Correlated positions
        if self._has_correlated_position(instrument, portfolio):
            return RiskCheckResult(approved=False, reason=f"Correlated position already open for {instrument}")
        checks_passed.append("correlation_ok")

        # 6. Economic news embargo
        if self._near_high_impact_event(calendar, self.limits.get("no_trade_before_news_minutes", 15)):
            return RiskCheckResult(approved=False, reason="High-impact event within embargo window")
        checks_passed.append("news_ok")

        # 7. Risk:Reward
        if signal.recommendation:
            rr = signal.recommendation.risk_reward_ratio
            min_rr = self.limits.get("min_risk_reward_ratio", 1.5)
            if rr > 0 and rr < min_rr:
                return RiskCheckResult(approved=False, reason=f"R:R {rr:.2f} below minimum {min_rr}")
        checks_passed.append("rr_ok")

        # 8. Max trades per day per instrument
        daily_count = self._today_trades.get(instrument, 0)
        if daily_count >= self.limits.get("max_trades_per_day_per_instrument", 10):
            return RiskCheckResult(approved=False, reason=f"Max daily trades ({daily_count}) reached for {instrument}")
        checks_passed.append("daily_count_ok")

        # 9. Min time between trades (soft — don't reject, just warn)
        last_trade = self._last_trade_time.get(instrument)
        min_gap_min = self.soft.get("min_minutes_between_trades", 5)
        if last_trade and (datetime.utcnow() - last_trade).total_seconds() / 60 < min_gap_min:
            log.warning("Trade within minimum gap", instrument=instrument, gap_min=min_gap_min)
        checks_passed.append("time_gap_ok")

        # 10. Consecutive loss reduction (soft)
        adjusted_size: float | None = None
        consecutive_losses = self._count_consecutive_losses()
        reduce_threshold = self.soft.get("max_consecutive_losses_reduce", 5)
        if consecutive_losses >= reduce_threshold:
            adjusted_size = self.soft.get("reduce_size_factor", 0.5)
            log.warning("Reducing size due to consecutive losses", count=consecutive_losses)

        return RiskCheckResult(
            approved=True,
            checks_passed=checks_passed,
            adjusted_size=adjusted_size,
        )

    def record_trade_executed(self, instrument: str) -> None:
        self._today_trades[instrument] = self._today_trades.get(instrument, 0) + 1
        self._last_trade_time[instrument] = datetime.utcnow()

    def _has_correlated_position(self, instrument: str, portfolio: PortfolioState) -> bool:
        group = _find_correlated_group(instrument)
        open_instruments = {p.instrument for p in portfolio.positions}
        correlated = group & open_instruments - {instrument}
        if not correlated:
            return False
        count = len(correlated) + 1
        return count > self.limits.get("max_correlated_positions", 3)

    def _near_high_impact_event(
        self, calendar: list[EconomicEvent], minutes: int = 15
    ) -> bool:
        threshold = datetime.utcnow() + timedelta(minutes=minutes)
        return any(e.impact == "HIGH" and e.timestamp <= threshold for e in calendar)

    def _count_consecutive_losses(self) -> int:
        if not self.store:
            return 0
        trades = self.store.get_trades(limit=20)
        count = 0
        for t in trades:
            if t.pnl < 0:
                count += 1
            else:
                break
        return count
