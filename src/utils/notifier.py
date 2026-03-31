"""Multi-channel notification system — Telegram + console fallback."""
from __future__ import annotations

import os

from src.utils.logger import get_logger

log = get_logger(__name__)


class Notifier:
    """Send trade alerts via Telegram (falls back to console log if not configured)."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.token = cfg.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = cfg.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", "")
        self._bot = None
        self._enabled = bool(self.token and self.chat_id)

    async def _get_bot(self):
        if self._bot is None and self._enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.token)
            except ImportError:
                log.warning("python-telegram-bot not installed, Telegram disabled")
                self._enabled = False
        return self._bot

    async def send(self, message: str, severity: str = "INFO") -> None:
        log.info("Alert", severity=severity, message=message[:100])
        bot = await self._get_bot()
        if bot:
            try:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode="HTML",
                )
            except Exception as exc:
                log.warning("Telegram send failed", error=str(exc))

    async def send_critical(self, message: str) -> None:
        await self.send(f"CRITICAL\n{message}", severity="CRITICAL")

    async def send_trade_open(self, instrument: str, side: str, price: float,
                               quantity: float, stop_loss: float | None,
                               take_profit: float | None, composite_score: float) -> None:
        sl_str = f"{stop_loss:.5g}" if stop_loss else "—"
        tp_str = f"{take_profit:.5g}" if take_profit else "—"
        msg = (
            f"OPEN {side} {instrument}\n"
            f"Price: {price:.5g}  Qty: {quantity:.4g}\n"
            f"SL: {sl_str}  TP: {tp_str}\n"
            f"Score: {composite_score:+.2f}"
        )
        await self.send(msg)

    async def send_trade_close(self, instrument: str, side: str, pnl: float,
                                pnl_pct: float, exit_reason: str,
                                duration_min: float) -> None:
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{'WIN' if pnl >= 0 else 'LOSS'} CLOSE {side} {instrument}\n"
            f"P&L: {sign}${pnl:.2f} ({sign}{pnl_pct:.2f}%)\n"
            f"Exit: {exit_reason}  Dur: {duration_min:.0f}min"
        )
        await self.send(msg)

    async def send_daily_summary(self, equity: float, daily_pnl: float,
                                  wins: int, losses: int,
                                  drawdown_pct: float) -> None:
        sign = "+" if daily_pnl >= 0 else ""
        msg = (
            f"Daily Summary\n"
            f"{'─' * 18}\n"
            f"Trades: {wins + losses} ({wins}W / {losses}L)\n"
            f"Daily P&L: {sign}${daily_pnl:,.2f}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Drawdown: {drawdown_pct:.1f}%"
        )
        await self.send(msg)
