"""Attribution engine — decompose trade P&L into contributing factors."""
from __future__ import annotations

from datetime import datetime

from src.types import Side, TradeRecord
from src.utils.logger import get_logger

log = get_logger(__name__)

# Factor weights for the composite attribution (must sum to 1.0)
FACTOR_WEIGHTS = {
    "technical":  0.30,
    "sentiment":  0.25,
    "regime":     0.20,
    "timing":     0.15,
    "sizing":     0.10,
}


class AttributionEngine:
    """Decompose each trade's P&L into factor contributions."""

    def __init__(self, store) -> None:
        self.store = store

    # --------------------------------------------------------- Single trade

    def attribute_trade(self, trade: TradeRecord) -> dict[str, float]:
        """
        Return a dict of factor → signed contribution score.
        Positive = factor helped, negative = factor hurt.
        Values are normalized so |values| sum to 1.0.
        """
        won = trade.pnl > 0
        direction = 1 if won else -1
        tech = trade.technical_at_entry or {}

        scores: dict[str, float] = {}

        # 1. Technical — did trend_bias align with the trade direction?
        trend_bias = tech.get("trend_bias", "neutral")
        if trade.side == Side.BUY:
            tech_aligned = trend_bias == "bullish"
        else:
            tech_aligned = trend_bias == "bearish"
        scores["technical"] = 1.0 if tech_aligned else -0.5

        # 2. Sentiment — was signal_score confident AND correct?
        conf = trade.signal_score  # composite_score, -1..+1
        if abs(conf) >= 0.4 and won:
            scores["sentiment"] = 0.8
        elif abs(conf) >= 0.4 and not won:
            scores["sentiment"] = -0.8   # high confidence wrong = bad signal
        else:
            scores["sentiment"] = 0.1 if won else -0.2

        # 3. Regime — was the regime directionally aligned?
        regime = trade.regime_at_entry or ""
        regime_bullish = regime in ("STRONG_TREND_UP", "WEAK_TREND", "BREAKOUT")
        regime_bearish = regime in ("STRONG_TREND_DOWN", "CRISIS")
        if trade.side == Side.BUY:
            regime_aligned = regime_bullish
        else:
            regime_aligned = regime_bearish
        if regime_aligned and won:
            scores["regime"] = 0.7
        elif not regime_aligned and not won:
            scores["regime"] = -0.7   # traded against regime, lost
        elif regime_aligned and not won:
            scores["regime"] = 0.2    # regime ok but still lost
        else:
            scores["regime"] = -0.3

        # 4. Timing — entry efficiency proxy
        #    Best case: price moved immediately in our favour (short duration + win)
        #    Worst case: held long, then stopped out
        dur = trade.duration_minutes
        if won and dur > 0:
            # Quick wins score higher on timing
            scores["timing"] = min(1.0, 60.0 / max(dur, 1)) * 0.8 + 0.2
        elif not won and dur < 30:
            scores["timing"] = -0.6   # stopped out quickly = poor timing
        elif not won:
            scores["timing"] = -0.3
        else:
            scores["timing"] = 0.5

        # 5. Sizing — was ATR-based sizing appropriate for realized move?
        atr = tech.get("atr_14", 0.0)
        if atr > 0:
            realized_move = abs(trade.exit_price - trade.entry_price)
            expected_move = atr * 2
            move_ratio = realized_move / expected_move
            if won:
                scores["sizing"] = min(1.0, move_ratio) * 0.6
            else:
                scores["sizing"] = -min(1.0, move_ratio) * 0.6
        else:
            scores["sizing"] = 0.1 if won else -0.1

        # Normalize magnitudes to sum to 1.0, preserve direction per factor
        total_mag = sum(abs(v) for v in scores.values())
        if total_mag > 0:
            normalized = {k: (abs(v) / total_mag) * (1 if v >= 0 else -1)
                          for k, v in scores.items()}
        else:
            normalized = {k: 0.0 for k in scores}

        log.debug("Trade attributed", trade_id=trade.id,
                  won=won, attribution=normalized)
        return normalized

    # -------------------------------------------------------- Period report

    def generate_report(self, start: datetime, end: datetime) -> dict:
        """Aggregate attribution across all trades in the period."""
        trades = self.store.get_trades(start=start, end=end)

        if not trades:
            return {"error": "No trades in period",
                    "period": {"start": start.isoformat(), "end": end.isoformat()}}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)

        # Run attribution for trades that lack it
        for trade in trades:
            if not trade.factor_attribution:
                trade.factor_attribution = self.attribute_trade(trade)
                try:
                    self.store.update_trade(trade)
                except Exception:
                    pass

        # Aggregate per factor
        factor_sums: dict[str, list[float]] = {}
        for trade in trades:
            for factor, value in trade.factor_attribution.items():
                factor_sums.setdefault(factor, []).append(value)
        avg_attribution = {k: sum(v) / len(v) for k, v in factor_sums.items()}

        sorted_factors = sorted(avg_attribution.items(), key=lambda x: x[1])
        best_factor = sorted_factors[-1][0] if sorted_factors else ""
        worst_factor = sorted_factors[0][0] if sorted_factors else ""

        # Per-instrument breakdown
        inst_breakdown: dict[str, dict] = {}
        for t in trades:
            b = inst_breakdown.setdefault(t.instrument, {"pnl": 0.0, "trades": 0, "wins": 0})
            b["pnl"] += t.pnl
            b["trades"] += 1
            b["wins"] += int(t.pnl > 0)

        # Per-regime breakdown
        regime_breakdown: dict[str, dict] = {}
        for t in trades:
            b = regime_breakdown.setdefault(t.regime_at_entry or "UNKNOWN",
                                            {"pnl": 0.0, "trades": 0, "wins": 0})
            b["pnl"] += t.pnl
            b["trades"] += 1
            b["wins"] += int(t.pnl > 0)

        # Lessons from Claude reflections
        lessons = [t.lessons_learned for t in trades if t.lessons_learned]
        unique_lessons = list(dict.fromkeys(lessons))[:5]

        report = {
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": len(wins) / len(trades) * 100,
            "total_pnl": round(total_pnl, 2),
            "avg_win_pct": (sum(t.pnl_pct for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss_pct": (sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0.0,
            "avg_attribution": {k: round(v, 3) for k, v in avg_attribution.items()},
            "best_factor": best_factor,
            "worst_factor": worst_factor,
            "instrument_breakdown": inst_breakdown,
            "regime_breakdown": regime_breakdown,
            "top_lessons": unique_lessons,
        }
        return report

    def print_report(self, report: dict) -> None:
        """Pretty-print attribution report to console."""
        if "error" in report:
            print(f"Attribution: {report['error']}")
            return

        p = report["period"]
        print(f"\n{'='*55}")
        print(f"  ATTRIBUTION REPORT  {p['start'][:10]} → {p['end'][:10]}")
        print(f"{'='*55}")
        print(f"  Trades:    {report['total_trades']:>4d}  "
              f"({report['winning_trades']}W / {report['losing_trades']}L  "
              f"{report['win_rate_pct']:.0f}%)")
        print(f"  Total P&L: ${report['total_pnl']:>10.2f}")
        print(f"  Avg Win:   {report['avg_win_pct']:>7.2f}%  "
              f"Avg Loss: {report['avg_loss_pct']:>7.2f}%")

        print(f"\n  Factor Attribution (avg signed contribution):")
        for factor, score in sorted(report["avg_attribution"].items(),
                                    key=lambda x: x[1], reverse=True):
            bar = "+" * int(abs(score) * 20) if score >= 0 else "-" * int(abs(score) * 20)
            sign = "+" if score >= 0 else ""
            print(f"    {factor:<12} {sign}{score:+.3f}  {bar}")

        print(f"\n  Best factor:  {report['best_factor']}")
        print(f"  Worst factor: {report['worst_factor']}")

        if report.get("instrument_breakdown"):
            print(f"\n  By Instrument:")
            for inst, data in sorted(report["instrument_breakdown"].items(),
                                     key=lambda x: x[1]["pnl"], reverse=True):
                wr = data["wins"] / data["trades"] * 100 if data["trades"] else 0
                print(f"    {inst:<12} P&L: ${data['pnl']:>8.2f}  "
                      f"({data['trades']} trades, {wr:.0f}% WR)")

        if report.get("regime_breakdown"):
            print(f"\n  By Regime:")
            for regime, data in sorted(report["regime_breakdown"].items(),
                                       key=lambda x: x[1]["pnl"], reverse=True):
                wr = data["wins"] / data["trades"] * 100 if data["trades"] else 0
                print(f"    {regime:<22} P&L: ${data['pnl']:>8.2f}  "
                      f"({data['trades']} trades, {wr:.0f}% WR)")

        if report.get("top_lessons"):
            print(f"\n  Lessons Learned:")
            for lesson in report["top_lessons"]:
                print(f"    - {lesson[:75]}")

        print(f"{'='*55}\n")
