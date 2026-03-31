"""PortfolioReport — cross-agent performance analytics and Phase 3 exit criteria."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.orchestrator.correlation_monitor import CorrelationMonitor
from src.utils.logger import get_logger
from src.utils.metrics import max_drawdown, sharpe_ratio

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

PHASE3_SHARPE_THRESHOLD = 2.0
PHASE3_CORR_THRESHOLD = 0.3


@dataclass
class AgentSummary:
    agent_id: str
    name: str
    total_trades: int
    win_rate_pct: float
    total_pnl: float
    sharpe_30d: float
    max_drawdown_pct: float
    fitness: float
    is_active: bool


@dataclass
class PortfolioSummary:
    generated_at: datetime
    total_trades: int
    total_pnl: float
    portfolio_sharpe: float          # combined equity curve Sharpe
    portfolio_max_dd: float
    avg_inter_agent_correlation: float  # from CorrelationMonitor
    diversification_score: float
    agent_summaries: list[AgentSummary]
    phase3_exit_met: bool            # Sharpe > 2.0 AND avg_corr < 0.3
    phase3_notes: list[str]          # human-readable criteria checks


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class PortfolioReport:
    """
    Generates a ``PortfolioSummary`` across all registered strategy agents
    and evaluates Phase 3 exit criteria.
    """

    def __init__(self, agents: list, correlation_monitor: CorrelationMonitor) -> None:
        """
        Parameters
        ----------
        agents : list[BaseStrategyAgent]
            All agents tracked by the system (active and inactive).
        correlation_monitor : CorrelationMonitor
            Shared monitor that owns inter-agent correlation logic.
        """
        self.agents = agents
        self.correlation_monitor = correlation_monitor

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        equity_curves: dict[str, list[float]],
        initial_capital: float,
    ) -> PortfolioSummary:
        """
        Build a ``PortfolioSummary`` from per-agent equity curves.

        Parameters
        ----------
        equity_curves : dict[str, list[float]]
            Mapping of agent_id → cumulative PnL equity curve (one value per
            time step, e.g. daily).  Agents missing from this dict are treated
            as flat (zero contribution).
        initial_capital : float
            Total starting capital across all agents; used as the denominator
            when computing portfolio-level returns.

        Returns
        -------
        PortfolioSummary
        """
        # ----------------------------------------------------------------
        # 1. Build combined portfolio equity curve (sum of individual PnLs)
        # ----------------------------------------------------------------
        portfolio_curve = self._combine_equity_curves(equity_curves)

        # ----------------------------------------------------------------
        # 2. Portfolio-level Sharpe
        # ----------------------------------------------------------------
        if len(portfolio_curve) >= 2 and initial_capital > 0:
            port_arr = np.array(portfolio_curve)
            port_returns = list(np.diff(port_arr) / initial_capital)
            portfolio_sharpe = sharpe_ratio(port_returns)
        else:
            portfolio_sharpe = 0.0

        # ----------------------------------------------------------------
        # 3. Portfolio max drawdown
        # ----------------------------------------------------------------
        if len(portfolio_curve) >= 2:
            # Shift curve to start at initial_capital so pct drawdown is valid
            shifted = [initial_capital + v for v in portfolio_curve]
            portfolio_max_dd = max_drawdown(shifted)
        else:
            portfolio_max_dd = 0.0

        # ----------------------------------------------------------------
        # 4. Correlation / diversification metrics
        # ----------------------------------------------------------------
        diversification_score = self.correlation_monitor.get_diversification_score()
        avg_inter_agent_correlation = self._compute_avg_correlation()

        # ----------------------------------------------------------------
        # 5. Per-agent summaries
        # ----------------------------------------------------------------
        agent_summaries: list[AgentSummary] = []
        total_trades = 0
        total_pnl = 0.0

        for agent in self.agents:
            p = agent.performance
            summary = AgentSummary(
                agent_id=agent.agent_id,
                name=agent.name,
                total_trades=p.total_trades,
                win_rate_pct=round(p.win_rate_pct, 2),
                total_pnl=round(p.total_pnl, 2),
                sharpe_30d=round(p.rolling_sharpe_30d, 4),
                max_drawdown_pct=round(p.max_drawdown_pct, 2),
                fitness=round(p.fitness, 4),
                is_active=agent.is_active,
            )
            agent_summaries.append(summary)
            total_trades += p.total_trades
            total_pnl += p.total_pnl

        # ----------------------------------------------------------------
        # 6. Phase 3 exit criteria
        # ----------------------------------------------------------------
        phase3_notes: list[str] = []

        sharpe_ok = portfolio_sharpe > PHASE3_SHARPE_THRESHOLD
        corr_ok = avg_inter_agent_correlation < PHASE3_CORR_THRESHOLD

        phase3_notes.append(
            f"Portfolio Sharpe {portfolio_sharpe:.3f} "
            f"{'>' if sharpe_ok else '<='} {PHASE3_SHARPE_THRESHOLD} "
            f"({'PASS' if sharpe_ok else 'FAIL'})"
        )
        phase3_notes.append(
            f"Avg inter-agent correlation {avg_inter_agent_correlation:.3f} "
            f"{'<' if corr_ok else '>='} {PHASE3_CORR_THRESHOLD} "
            f"({'PASS' if corr_ok else 'FAIL'})"
        )

        phase3_exit_met = sharpe_ok and corr_ok
        if phase3_exit_met:
            phase3_notes.append("ALL criteria met — Phase 3 exit conditions satisfied.")
        else:
            phase3_notes.append("Phase 3 exit conditions NOT yet met.")

        log.info(
            "Portfolio report generated",
            agents=len(agent_summaries),
            total_trades=total_trades,
            portfolio_sharpe=round(portfolio_sharpe, 4),
            portfolio_max_dd=round(portfolio_max_dd, 2),
            avg_corr=round(avg_inter_agent_correlation, 4),
            phase3_exit_met=phase3_exit_met,
        )

        return PortfolioSummary(
            generated_at=datetime.utcnow(),
            total_trades=total_trades,
            total_pnl=round(total_pnl, 2),
            portfolio_sharpe=round(portfolio_sharpe, 4),
            portfolio_max_dd=round(portfolio_max_dd, 2),
            avg_inter_agent_correlation=round(avg_inter_agent_correlation, 4),
            diversification_score=round(diversification_score, 4),
            agent_summaries=agent_summaries,
            phase3_exit_met=phase3_exit_met,
            phase3_notes=phase3_notes,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(summary: PortfolioSummary) -> None:
        """Print a formatted ASCII report to stdout."""
        width = 105
        sep = "=" * width
        thin = "-" * width

        print(sep)
        print(f"  ATLAS-TRADER  |  PORTFOLIO PERFORMANCE REPORT")
        print(f"  Generated: {summary.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(sep)

        # ---- Per-agent table ----------------------------------------
        hdr = (
            f"  {'ID':<22}  {'Name':<22}  {'Trades':>6}  {'Win%':>6}  "
            f"{'PnL':>10}  {'Sharpe':>7}  {'MaxDD%':>7}  {'Fitness':>7}  {'Active':>6}"
        )
        print(hdr)
        print(thin)

        for a in summary.agent_summaries:
            active_str = "yes" if a.is_active else "no"
            print(
                f"  {a.agent_id:<22}  {a.name:<22}  {a.total_trades:>6}  "
                f"{a.win_rate_pct:>5.1f}%  {a.total_pnl:>10.2f}  "
                f"{a.sharpe_30d:>7.3f}  {a.max_drawdown_pct:>6.2f}%  "
                f"{a.fitness:>7.4f}  {active_str:>6}"
            )

        print(thin)

        # ---- Portfolio-level metrics --------------------------------
        print()
        print("  PORTFOLIO METRICS")
        print(thin)
        print(f"  {'Total Trades':<35} {summary.total_trades:>10}")
        print(f"  {'Total PnL':<35} {summary.total_pnl:>10.2f}")
        print(f"  {'Portfolio Sharpe':<35} {summary.portfolio_sharpe:>10.4f}")
        print(f"  {'Portfolio Max Drawdown':<35} {summary.portfolio_max_dd:>9.2f}%")
        print(f"  {'Avg Inter-Agent Correlation':<35} {summary.avg_inter_agent_correlation:>10.4f}")
        print(f"  {'Diversification Score':<35} {summary.diversification_score:>10.4f}")
        print()

        # ---- Phase 3 exit criteria ----------------------------------
        print("  PHASE 3 EXIT CRITERIA")
        print(thin)
        for note in summary.phase3_notes:
            # Choose tick or cross based on PASS/FAIL in note text
            if "PASS" in note or "ALL criteria met" in note:
                marker = "\u2713"
            else:
                marker = "\u2717"
            print(f"  {marker}  {note}")
        print()

        overall = "MET" if summary.phase3_exit_met else "NOT MET"
        print(f"  >> Phase 3 Exit: {overall}")
        print(sep)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _combine_equity_curves(
        self, equity_curves: dict[str, list[float]]
    ) -> list[float]:
        """
        Sum per-agent equity curves into a single portfolio curve.

        Curves of different lengths are zero-padded on the left so that the
        combined curve always extends to the longest individual curve.
        """
        if not equity_curves:
            return []

        max_len = max(len(v) for v in equity_curves.values())
        if max_len == 0:
            return []

        combined = np.zeros(max_len, dtype=float)
        for curve in equity_curves.values():
            arr = np.array(curve, dtype=float)
            # Pad shorter curves with their first value on the left (or 0 if empty)
            pad = max_len - len(arr)
            if pad > 0:
                arr = np.concatenate([np.zeros(pad), arr])
            combined += arr

        return combined.tolist()

    def _compute_avg_correlation(self) -> float:
        """
        Derive the average pairwise |correlation| from the correlation monitor.

        Transforms the diversification score back to an average correlation:
            avg_corr = 1 - diversification_score
        This avoids re-computing the correlation matrix here.
        """
        div_score = self.correlation_monitor.get_diversification_score()
        # get_diversification_score returns 1.0 - avg_abs_corr, so invert it.
        # Clamp to [0, 1] to guard against floating-point noise.
        return float(np.clip(1.0 - div_score, 0.0, 1.0))
