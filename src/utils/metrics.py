"""Performance metric calculators."""
from __future__ import annotations
import math
import numpy as np


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free_rate / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: list[float]) -> float:
    """Returns max drawdown as a positive percentage (0-100)."""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(-np.min(drawdown) * 100)


def win_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls) * 100


def profit_factor(pnls: list[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calmar_ratio(annual_return_pct: float, max_dd_pct: float) -> float:
    if max_dd_pct == 0:
        return 0.0
    return annual_return_pct / max_dd_pct
