"""Trade journal — full context capture at open/close + Claude reflection."""
from __future__ import annotations

import dataclasses
from datetime import datetime

from src.types import (
    MarketRegime, NewsItem, Order, PortfolioState, SignalScore,
    TechnicalSnapshot, TradeRecord,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class TradeJournal:
    def __init__(self, store, claude_agent=None, notifier=None) -> None:
        self.store = store
        self.claude = claude_agent
        self.notifier = notifier

    # ---------------------------------------------------------------- Open

    async def record_trade_open(
        self,
        order: Order,
        signal: SignalScore,
        technical: TechnicalSnapshot,
        regime: MarketRegime,
        news: list[NewsItem],
    ) -> None:
        """Snapshot everything at trade entry and persist to journal."""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "instrument": order.instrument,
            "side": order.side.value,
            "entry_price": order.fill_price,
            "quantity": order.fill_quantity,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            # Signal scores
            "composite_score": signal.composite_score,
            "technical_score": signal.technical_score,
            "sentiment_score": signal.sentiment_score,
            "regime_score": signal.regime_score,
            "claude_confidence": signal.claude_confidence,
            "action": signal.action.value,
            # Market context
            "regime": regime.value,
            "indicators": technical.indicators,
            "signals": technical.signals,
            "trend_bias": technical.trend_bias,
            "news_headlines": [n.title for n in news[:5]],
            "claude_reasoning": order.claude_reasoning,
        }
        self.store.save_journal_entry("open", order.id, order.instrument, context)

        if self.notifier:
            await self.notifier.send_trade_open(
                instrument=order.instrument,
                side=order.side.value,
                price=order.fill_price or 0.0,
                quantity=order.fill_quantity,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                composite_score=signal.composite_score,
            )

        log.info("Trade open journalled", id=order.id, instrument=order.instrument,
                 side=order.side.value, score=round(signal.composite_score, 3))

    # --------------------------------------------------------------- Close

    async def record_trade_close(
        self,
        trade: TradeRecord,
        exit_technical: TechnicalSnapshot | None = None,
        exit_regime: MarketRegime | None = None,
    ) -> TradeRecord:
        """Capture exit context, trigger Claude reflection, save trade."""

        # Persist exit context
        exit_context = {
            "timestamp": datetime.utcnow().isoformat(),
            "exit_price": trade.exit_price,
            "pnl": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "exit_reason": trade.exit_reason.value,
            "duration_minutes": trade.duration_minutes,
            "regime": exit_regime.value if exit_regime else "",
            "indicators": exit_technical.indicators if exit_technical else {},
        }
        self.store.save_journal_entry("close", trade.id, trade.instrument, exit_context)

        # Claude reflection (non-blocking: failure just means no attribution)
        if self.claude:
            try:
                entry_ctx = self.store.get_journal_entry("open", trade.id)
                reflection = await self.claude.reflect_on_trade(trade, str(entry_ctx))
                if reflection:
                    trade.factor_attribution = reflection.get("factor_attribution", {})
                    trade.lessons_learned = reflection.get("lessons", "")
                    trade.claude_reasoning_exit = reflection.get("summary", "")
                    grade = reflection.get("performance_grade", "?")
                    log.info("Reflection done", id=trade.id, grade=grade,
                             lessons=trade.lessons_learned[:80])
            except Exception as exc:
                log.warning("Reflection failed", trade_id=trade.id, error=str(exc))

        # Save trade record
        self.store.save_trade(trade)
        log.info("Trade close journalled",
                 id=trade.id, instrument=trade.instrument,
                 pnl=round(trade.pnl, 2), pnl_pct=round(trade.pnl_pct, 2),
                 reason=trade.exit_reason.value, duration_min=round(trade.duration_minutes))

        if self.notifier:
            await self.notifier.send_trade_close(
                instrument=trade.instrument,
                side=trade.side.value,
                pnl=trade.pnl,
                pnl_pct=trade.pnl_pct,
                exit_reason=trade.exit_reason.value,
                duration_min=trade.duration_minutes,
            )

        return trade

    # -------------------------------------------------------- Legacy method

    async def record_trade(
        self,
        trade: TradeRecord,
        portfolio: PortfolioState,
        entry_context: str = "",
    ) -> TradeRecord:
        """Backward-compatible: enrich trade with Claude reflection then save."""
        if self.claude:
            try:
                reflection = await self.claude.reflect_on_trade(trade, entry_context)
                if reflection:
                    trade.factor_attribution = reflection.get("factor_attribution", {})
                    trade.lessons_learned = reflection.get("lessons", "")
                    grade = reflection.get("performance_grade", "")
                    log.info("Trade reflected", id=trade.id, grade=grade)
            except Exception as exc:
                log.warning("Reflection failed", error=str(exc))

        self.store.save_trade(trade)
        log.info("Trade recorded", id=trade.id, instrument=trade.instrument,
                 pnl=round(trade.pnl, 2), reason=trade.exit_reason.value)
        return trade

    # ------------------------------------------------------------ Summary

    def print_summary(self, trades: list[TradeRecord]) -> None:
        if not trades:
            print("No trades yet.")
            return

        wins = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) * 100

        print(f"\n{'='*50}")
        print(f"  TRADE JOURNAL SUMMARY ({len(trades)} trades)")
        print(f"{'='*50}")
        print(f"  Total P&L:    ${total_pnl:>10.2f}")
        print(f"  Win Rate:     {win_rate:>9.1f}%")
        print(f"  Winners:      {len(wins):>10d}")
        print(f"  Losers:       {len(losers):>10d}")
        if wins:
            print(f"  Avg Win:      {sum(t.pnl_pct for t in wins)/len(wins):>9.2f}%")
        if losers:
            print(f"  Avg Loss:     {sum(t.pnl_pct for t in losers)/len(losers):>9.2f}%")

        # Top lessons from Claude reflections
        lessons = [t.lessons_learned for t in trades if t.lessons_learned]
        if lessons:
            print(f"\n  Recent lessons:")
            for lesson in lessons[-3:]:
                print(f"    - {lesson[:80]}")
        print(f"{'='*50}\n")
