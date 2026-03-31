"""Paper trading engine — realistic fill simulation."""
from __future__ import annotations
import asyncio
from datetime import datetime

import numpy as np

from src.types import (
    Bar, ExitReason, Order, OrderStatus, OrderType, PortfolioState,
    Position, Side, TradeRecord,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class PaperTradingEngine:
    def __init__(self, config: dict, store=None) -> None:
        self.slippage_mean = config.get("slippage_mean_pct", 0.02) / 100
        self.slippage_std = config.get("slippage_std_pct", 0.01) / 100
        self.latency_ms = config.get("latency_ms", 50)
        self.commission_pct = config.get("commission_pct", 0.1) / 100
        self.store = store

        initial_capital = config.get("initial_capital", 100_000.0)
        self.portfolio = PortfolioState(
            equity=initial_capital,
            cash=initial_capital,
            total_unrealized_pnl=0.0,
            total_realized_pnl=0.0,
        )
        self.equity_high_water = initial_capital
        self._day_start_equity = initial_capital

    # ---------------------------------------------------------------- Orders

    async def execute_order(self, order: Order, current_bar: Bar) -> Order:
        """Simulate order fill with realistic slippage and latency."""
        await asyncio.sleep(self.latency_ms / 1000)

        if order.order_type == OrderType.MARKET:
            slippage = np.random.normal(self.slippage_mean, self.slippage_std)
            slippage = max(0, slippage)  # slippage always costs you
            if order.side == Side.BUY:
                fill_price = current_bar.close * (1 + slippage)
            else:
                fill_price = current_bar.close * (1 - slippage)

            order.fill_price = round(fill_price, 6)
            order.slippage = round(abs(fill_price - current_bar.close), 6)
            order.fill_quantity = order.quantity
            order.filled_at = datetime.utcnow()
            order.status = OrderStatus.FILLED

        elif order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY and order.price and current_bar.low <= order.price:
                order.fill_price = order.price
                order.status = OrderStatus.FILLED
            elif order.side == Side.SELL and order.price and current_bar.high >= order.price:
                order.fill_price = order.price
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PENDING
                return order

            order.filled_at = datetime.utcnow()
            order.fill_quantity = order.quantity
            order.slippage = 0.0

        if order.status == OrderStatus.FILLED:
            order.commission = self._calc_commission(order)
            self._open_position(order)

        return order

    def _calc_commission(self, order: Order) -> float:
        if order.fill_price and order.fill_quantity:
            return order.fill_price * order.fill_quantity * self.commission_pct
        return 0.0

    def _open_position(self, order: Order) -> None:
        if order.fill_price is None:
            return
        pos = Position(
            instrument=order.instrument,
            side=order.side,
            quantity=order.fill_quantity,
            entry_price=order.fill_price,
            current_price=order.fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            order_ids=[order.id],
        )
        self.portfolio.positions.append(pos)
        cost = order.fill_price * order.fill_quantity + order.commission
        if order.side == Side.BUY:
            self.portfolio.cash -= cost
        # Recalculate equity immediately: cash + market value of all positions
        self.portfolio.equity = self.portfolio.cash + sum(
            p.current_price * p.quantity for p in self.portfolio.positions
        )
        log.info("Position opened", instrument=order.instrument, side=order.side.value,
                 qty=order.fill_quantity, price=order.fill_price)

    # ----------------------------------------------------------------- Stops

    def check_stops(self, current_bar: Bar) -> list[TradeRecord]:
        """Check all positions for SL/TP hits. Returns closed trade records."""
        closed_trades: list[TradeRecord] = []
        remaining: list[Position] = []

        for pos in self.portfolio.positions:
            if pos.instrument != current_bar.instrument:
                remaining.append(pos)
                continue

            exit_price: float | None = None
            exit_reason: ExitReason | None = None

            if pos.stop_loss:
                if pos.side == Side.BUY and current_bar.low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = ExitReason.STOP_LOSS
                elif pos.side == Side.SELL and current_bar.high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = ExitReason.STOP_LOSS

            if exit_price is None and pos.take_profit:
                if pos.side == Side.BUY and current_bar.high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = ExitReason.TAKE_PROFIT
                elif pos.side == Side.SELL and current_bar.low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = ExitReason.TAKE_PROFIT

            if exit_price is not None and exit_reason is not None:
                trade = self._close_position(pos, exit_price, exit_reason)
                closed_trades.append(trade)
                if self.store:
                    self.store.save_trade(trade)
            else:
                remaining.append(pos)

        self.portfolio.positions = remaining
        return closed_trades

    def _close_position(
        self, pos: Position, exit_price: float, reason: ExitReason
    ) -> TradeRecord:
        import uuid
        if pos.side == Side.BUY:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        commission = exit_price * pos.quantity * self.commission_pct
        pnl -= commission

        duration = (datetime.utcnow() - pos.opened_at).total_seconds() / 60
        pnl_pct = pnl / (pos.entry_price * pos.quantity) * 100

        self.portfolio.cash += exit_price * pos.quantity - commission
        self.portfolio.total_realized_pnl += pnl

        trade = TradeRecord(
            id=f"TRD-{uuid.uuid4().hex[:8]}",
            instrument=pos.instrument,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            commission_total=round(commission, 4),
            entry_time=pos.opened_at,
            exit_time=datetime.utcnow(),
            duration_minutes=round(duration, 1),
            exit_reason=reason,
        )
        log.info("Position closed", instrument=pos.instrument,
                 pnl=round(pnl, 2), reason=reason.value)
        return trade

    def force_close_all(self, current_prices: dict[str, float]) -> list[TradeRecord]:
        """Close all positions at current market price (session end)."""
        trades: list[TradeRecord] = []
        for pos in self.portfolio.positions[:]:
            price = current_prices.get(pos.instrument, pos.current_price)
            trade = self._close_position(pos, price, ExitReason.SESSION_END)
            trades.append(trade)
            if self.store:
                self.store.save_trade(trade)
        self.portfolio.positions = []
        return trades

    # --------------------------------------------------------- Mark-to-Market

    def update_mark_to_market(self, current_prices: dict[str, float]) -> None:
        total_unrealized = 0.0
        total_position_value = 0.0
        for pos in self.portfolio.positions:
            price = current_prices.get(pos.instrument, pos.current_price)
            pos.current_price = price
            if pos.side == Side.BUY:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
            total_unrealized += pos.unrealized_pnl
            total_position_value += price * pos.quantity

        self.portfolio.total_unrealized_pnl = total_unrealized
        # equity = liquid cash + current market value of all positions
        self.portfolio.equity = self.portfolio.cash + total_position_value

        # Drawdown
        if self.portfolio.equity > self.equity_high_water:
            self.equity_high_water = self.portfolio.equity
        if self.equity_high_water > 0:
            dd = (self.equity_high_water - self.portfolio.equity) / self.equity_high_water * 100
            self.portfolio.max_drawdown_current = max(0.0, dd)

        # Daily P&L
        self.portfolio.daily_pnl = self.portfolio.equity - self._day_start_equity

        # Portfolio heat (total risk at stop)
        total_risk = sum(
            abs(p.entry_price - p.stop_loss) * p.quantity
            for p in self.portfolio.positions
            if p.stop_loss
        )
        self.portfolio.portfolio_heat_pct = (
            total_risk / self.portfolio.equity * 100
            if self.portfolio.equity > 0 else 0.0
        )
        self.portfolio.timestamp = datetime.utcnow()

    def reset_daily_pnl(self) -> None:
        self._day_start_equity = self.portfolio.equity
        self.portfolio.daily_pnl = 0.0
