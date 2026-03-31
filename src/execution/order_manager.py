"""Order lifecycle manager — routing, retry, position sync, kill switch."""
from __future__ import annotations

import asyncio
from datetime import datetime

from src.execution.exchange_adapter import AdapterFactory, ExchangeAdapter
from src.types import Order, OrderResult, OrderStatus, OrderType, Position, Side
from src.utils.logger import get_logger

log = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0   # seconds


class KillSwitch:
    """Hard stop: liquidate everything when drawdown breaches limit."""

    def __init__(self, max_drawdown_pct: float, max_daily_loss_pct: float) -> None:
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.active = False

    async def check(self, drawdown_pct: float, daily_loss_pct: float,
                    notifier=None) -> bool:
        """Return True (and arm) if any limit is breached."""
        if self.active:
            return True

        reason = None
        if drawdown_pct >= self.max_drawdown_pct:
            reason = f"Max drawdown {drawdown_pct:.1f}% >= {self.max_drawdown_pct}%"
        elif daily_loss_pct >= self.max_daily_loss_pct:
            reason = f"Daily loss {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%"

        if reason:
            self.active = True
            log.critical("KILL SWITCH ARMED", reason=reason)
            if notifier:
                await notifier.send_critical(f"KILL SWITCH: {reason}")
            await AdapterFactory.close_all_positions()
            return True

        return False

    def reset(self) -> None:
        """Manual reset (requires operator confirmation)."""
        self.active = False
        log.warning("Kill switch manually reset")


class OrderManager:
    """Manages the full lifecycle of orders across all exchanges."""

    def __init__(self, config: dict, store=None, risk_manager=None) -> None:
        self.config = config
        self.store = store
        self.risk_manager = risk_manager
        self._pending: dict[str, Order] = {}

    def _get_adapter(self, instrument: str) -> ExchangeAdapter:
        return AdapterFactory.get_adapter(instrument, self.config)

    def _smart_route(self, order: Order) -> Order:
        """Convert market orders to aggressive IOC limit when possible to reduce slippage."""
        if order.order_type == OrderType.MARKET and order.price:
            # Add 5 bps buffer above ask for buys, below bid for sells
            buffer = order.price * 0.0005
            order.order_type = OrderType.LIMIT
            if order.side == Side.BUY:
                order.price = round(order.price + buffer, 5)
            else:
                order.price = round(order.price - buffer, 5)
            order.time_in_force = "IOC"
        return order

    async def submit(self, order: Order, use_smart_routing: bool = True) -> OrderResult:
        """Submit order with retry logic."""
        adapter = self._get_adapter(order.instrument)

        if use_smart_routing:
            order = self._smart_route(order)

        last_result: OrderResult | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = await adapter.place_order(order)
                if result.status != OrderStatus.REJECTED:
                    order.status = result.status
                    order.submitted_at = datetime.utcnow()
                    if result.fill_price:
                        order.fill_price = result.fill_price
                        order.fill_quantity = result.fill_quantity
                        order.commission = result.commission
                        order.filled_at = datetime.utcnow()
                    self._pending[order.id] = order
                    if self.store:
                        try:
                            self.store.save_order(order)
                        except AttributeError:
                            pass  # DataStore may not have save_order yet
                    log.info("Order submitted", id=order.id, instrument=order.instrument,
                             status=result.status.value, attempt=attempt + 1)
                    return result

                # Don't retry on insufficient funds
                if result.message and "insufficient" in result.message.lower():
                    log.error("Insufficient funds", order_id=order.id)
                    return result

                last_result = result
                log.warning("Order rejected, retrying", attempt=attempt + 1,
                            order_id=order.id, msg=result.message)

            except Exception as exc:
                log.warning("Order attempt exception", attempt=attempt + 1, error=str(exc))
                last_result = OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                                          message=str(exc))

            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_RETRY_BACKOFF * (attempt + 1))

        log.error("Order failed after retries", order_id=order.id)
        return last_result or OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                                          message="Max retries exceeded")

    async def cancel(self, order_id: str, instrument: str) -> bool:
        adapter = self._get_adapter(instrument)
        success = await adapter.cancel_order(order_id)
        if success:
            self._pending.pop(order_id, None)
        return success

    async def sync_positions(self) -> dict[str, list[Position]]:
        """Fetch live positions from all connected adapters."""
        all_positions: dict[str, list[Position]] = {}
        for name, adapter in AdapterFactory._adapters.items():
            try:
                positions = await adapter.get_positions()
                all_positions[name] = positions
            except Exception as exc:
                log.error("Position sync failed", adapter=name, error=str(exc))
                all_positions[name] = []
        return all_positions
