"""Base class for all specialist strategy agents (Phase 3)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import polars as pl

from src.types import (
    MarketRegime, NewsItem, SignalScore, TechnicalSnapshot, TradeRecord,
)
from src.utils.logger import get_logger
from src.utils.metrics import sharpe_ratio, max_drawdown, win_rate, profit_factor

log = get_logger(__name__)


@dataclass
class AgentPerformance:
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    rolling_sharpe_30d: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 1.0
    avg_trade_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    daily_returns: list[float] = field(default_factory=list)

    def update(self, trades: list[TradeRecord], initial_capital: float = 100_000) -> None:
        if not trades:
            return
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.pnl > 0)
        self.total_pnl = sum(t.pnl for t in trades)
        self.avg_trade_pnl = self.total_pnl / self.total_trades

        pnls = [t.pnl for t in trades]
        self.win_rate_pct = win_rate(pnls)
        self.profit_factor = profit_factor(pnls)

        # 30-day rolling Sharpe
        cutoff = datetime.utcnow().timestamp() - 30 * 86400
        recent = [t for t in trades if t.exit_time.timestamp() >= cutoff]
        if recent:
            returns = [t.pnl / initial_capital for t in recent]
            self.rolling_sharpe_30d = sharpe_ratio(returns)

        # Equity curve for drawdown (simplified: cumulative PnL)
        cum = np.cumsum(pnls).tolist()
        self.max_drawdown_pct = max_drawdown(cum)
        self.last_updated = datetime.utcnow()

    @property
    def fitness(self) -> float:
        """Composite fitness 0..1 for capital allocation."""
        sharpe_norm = min(1.0, max(0.0, self.rolling_sharpe_30d / 3.0))
        wr_norm = self.win_rate_pct / 100.0
        dd_penalty = max(0.0, 1.0 - self.max_drawdown_pct / 20.0)
        return (sharpe_norm * 0.5 + wr_norm * 0.3 + dd_penalty * 0.2)


class BaseStrategyAgent(ABC):
    """Abstract base for all specialist trading agents."""

    def __init__(self, agent_id: str, config: dict, store, claude_agent,
                 technical) -> None:
        self.agent_id = agent_id
        self.config = config
        self.store = store
        self.claude = claude_agent
        self.technical = technical
        self.is_active = True
        self.allocated_capital = 0.0
        self.performance = AgentPerformance()

    # --- Abstract properties ---

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def preferred_regimes(self) -> list[MarketRegime]: ...

    @property
    @abstractmethod
    def target_instruments(self) -> list[str]: ...

    # --- Abstract methods ---

    @abstractmethod
    async def generate_signals(
        self,
        market_data: dict[str, pl.DataFrame],
        technical: dict[str, TechnicalSnapshot],
        news: list[NewsItem],
        regime: MarketRegime,
    ) -> list[SignalScore]: ...

    @abstractmethod
    def get_parameters(self) -> dict: ...

    @abstractmethod
    def set_parameters(self, params: dict) -> None: ...

    # --- Concrete helpers ---

    async def evaluate_fitness(self, regime: MarketRegime) -> float:
        """How well-suited is this agent for the current regime? 0.0-1.0"""
        base = self.performance.fitness
        regime_bonus = 0.2 if regime in self.preferred_regimes else 0.0
        return min(1.0, base + regime_bonus)

    def update_performance(self, initial_capital: float = 100_000) -> None:
        trades = self.store.get_trades(limit=500)
        self.performance.update(trades, initial_capital)
        log.debug("Agent performance updated",
                  agent=self.agent_id,
                  trades=self.performance.total_trades,
                  sharpe=round(self.performance.rolling_sharpe_30d, 2))

    def status_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "active": self.is_active,
            "allocated_capital": self.allocated_capital,
            "total_trades": self.performance.total_trades,
            "total_pnl": self.performance.total_pnl,
            "win_rate": self.performance.win_rate_pct,
            "sharpe_30d": self.performance.rolling_sharpe_30d,
            "max_dd": self.performance.max_drawdown_pct,
            "fitness": self.performance.fitness,
            "preferred_regimes": [r.value for r in self.preferred_regimes],
        }
