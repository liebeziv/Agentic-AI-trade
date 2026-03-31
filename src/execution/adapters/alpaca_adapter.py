"""Alpaca adapter for US stocks — supports paper and live trading."""
from __future__ import annotations

import os
from datetime import datetime

from src.execution.exchange_adapter import ExchangeAdapter
from src.types import (
    AccountInfo, Order, OrderResult, OrderStatus, OrderType,
    Position, Side,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def _map_alpaca_status(status: str) -> OrderStatus:
    mapping = {
        "new": OrderStatus.SUBMITTED,
        "partially_filled": OrderStatus.PARTIAL_FILL,
        "filled": OrderStatus.FILLED,
        "done_for_day": OrderStatus.CANCELLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "replaced": OrderStatus.CANCELLED,
        "pending_cancel": OrderStatus.SUBMITTED,
        "pending_replace": OrderStatus.SUBMITTED,
        "accepted": OrderStatus.SUBMITTED,
        "pending_new": OrderStatus.PENDING,
        "accepted_for_bidding": OrderStatus.SUBMITTED,
        "stopped": OrderStatus.CANCELLED,
        "rejected": OrderStatus.REJECTED,
        "suspended": OrderStatus.REJECTED,
        "calculated": OrderStatus.SUBMITTED,
    }
    return mapping.get(status, OrderStatus.PENDING)


class AlpacaAdapter(ExchangeAdapter):
    """Alpaca Markets adapter (paper by default)."""

    def __init__(self, config: dict) -> None:
        self.api_key = config.get("api_key") or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = config.get("secret_key") or os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = config.get("paper", True)
        self._client = None
        self._connected = False

    async def connect(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError("alpaca-py not installed: pip install alpaca-py")

        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        self._client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )
        account = self._client.get_account()
        if str(account.status) not in ("ACTIVE", "AccountStatus.ACTIVE"):
            raise ConnectionError(f"Alpaca account not active: {account.status}")
        self._connected = True
        log.info("Alpaca connected", paper=self.paper, account_id=str(account.id))

    async def disconnect(self) -> None:
        self._connected = False
        self._client = None

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def supports_instrument(self, instrument: str) -> bool:
        return "/" not in instrument and not instrument.endswith(".HK")

    async def get_account(self) -> AccountInfo:
        acc = self._client.get_account()
        return AccountInfo(
            equity=float(acc.equity),
            cash=float(acc.cash),
            buying_power=float(acc.buying_power),
            margin_used=float(acc.initial_margin) if acc.initial_margin else 0.0,
            currency="USD",
        )

    async def place_order(self, order: Order) -> OrderResult:
        try:
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError("alpaca-py not installed")

        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

        try:
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.instrument,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT and order.price:
                request = LimitOrderRequest(
                    symbol=order.instrument,
                    qty=order.quantity,
                    side=side,
                    limit_price=order.price,
                    time_in_force=TimeInForce.GTC,
                )
            else:
                return OrderResult(
                    order_id=order.id, status=OrderStatus.REJECTED,
                    message="Unsupported order type",
                )

            result = self._client.submit_order(request)
            fill_price = float(result.filled_avg_price) if result.filled_avg_price else None
            fill_qty = float(result.filled_qty) if result.filled_qty else 0.0

            return OrderResult(
                order_id=order.id,
                status=_map_alpaca_status(str(result.status).split(".")[-1].lower()),
                fill_price=fill_price,
                fill_quantity=fill_qty,
                message=str(result.id),  # exchange order id stored in message
            )
        except Exception as exc:
            log.warning("Alpaca order failed", instrument=order.instrument, error=str(exc))
            return OrderResult(
                order_id=order.id, status=OrderStatus.REJECTED, message=str(exc)
            )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            log.warning("Alpaca cancel failed", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            order = self._client.get_order_by_id(order_id)
            return _map_alpaca_status(str(order.status).split(".")[-1].lower())
        except Exception:
            return OrderStatus.REJECTED

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbol=instrument)
        orders = self._client.get_orders(filter=request)
        result = []
        for o in orders:
            result.append(Order(
                id=str(o.client_order_id or o.id),
                instrument=o.symbol,
                side=Side.BUY if str(o.side) in ("buy", "OrderSide.BUY") else Side.SELL,
                order_type=OrderType.MARKET if str(o.type) in ("market", "OrderType.MARKET") else OrderType.LIMIT,
                quantity=float(o.qty),
                price=float(o.limit_price) if o.limit_price else None,
                status=_map_alpaca_status(str(o.status).split(".")[-1].lower()),
            ))
        return result

    async def get_positions(self) -> list[Position]:
        positions = self._client.get_all_positions()
        result = []
        for p in positions:
            qty = float(p.qty)
            result.append(Position(
                instrument=p.symbol,
                side=Side.BUY if qty > 0 else Side.SELL,
                quantity=abs(qty),
                entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price) if p.current_price else 0.0,
                unrealized_pnl=float(p.unrealized_pl) if p.unrealized_pl else 0.0,
            ))
        return result

    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        try:
            if quantity:
                self._client.close_position(instrument, qty=quantity)
            else:
                self._client.close_position(instrument)
            return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.SUBMITTED)
        except Exception as exc:
            return OrderResult(
                order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED, message=str(exc)
            )

    async def close_all_positions(self) -> list[OrderResult]:
        try:
            self._client.close_all_positions(cancel_orders=True)
            log.info("Alpaca: all positions closed")
            return [OrderResult(order_id="CLOSE_ALL", status=OrderStatus.SUBMITTED)]
        except Exception as exc:
            log.error("Alpaca close_all failed", error=str(exc))
            return [OrderResult(order_id="CLOSE_ALL", status=OrderStatus.REJECTED, message=str(exc))]
