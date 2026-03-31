"""MetaTrader 5 adapter for forex (Windows only)."""
from __future__ import annotations

import asyncio
import os

from src.execution.exchange_adapter import ExchangeAdapter
from src.types import (
    AccountInfo, Order, OrderResult, OrderStatus, OrderType,
    Position, Side,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_FOREX_PAIRS = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF",
    "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
}


class MT5Adapter(ExchangeAdapter):
    """MT5 adapter for forex trading (requires MT5 terminal installed)."""

    def __init__(self, config: dict) -> None:
        self.login = int(config.get("login") or os.getenv("MT5_LOGIN", 0))
        self.password = config.get("password") or os.getenv("MT5_PASSWORD", "")
        self.server = config.get("server") or os.getenv("MT5_SERVER", "")
        self._mt5 = None
        self._connected = False

    async def connect(self) -> None:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            raise ImportError("MetaTrader5 not installed: pip install MetaTrader5")

        if not self._mt5.initialize():
            raise ConnectionError(f"MT5 init failed: {self._mt5.last_error()}")

        if self.login and self.password and self.server:
            authorized = self._mt5.login(self.login, password=self.password, server=self.server)
            if not authorized:
                raise ConnectionError(f"MT5 login failed: {self._mt5.last_error()}")

        info = self._mt5.account_info()
        self._connected = True
        log.info("MT5 connected", server=self.server,
                 account=info.login if info else "unknown",
                 balance=info.balance if info else 0)

    async def disconnect(self) -> None:
        if self._mt5:
            self._mt5.shutdown()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._mt5 is not None

    def supports_instrument(self, instrument: str) -> bool:
        return instrument in _FOREX_PAIRS or instrument.endswith("USD") or instrument.endswith("JPY")

    async def get_account(self) -> AccountInfo:
        info = self._mt5.account_info()
        if not info:
            return AccountInfo(equity=0, cash=0, buying_power=0)
        return AccountInfo(
            equity=float(info.equity),
            cash=float(info.balance),
            buying_power=float(info.margin_free),
            margin_used=float(info.margin),
            currency=info.currency,
        )

    async def place_order(self, order: Order) -> OrderResult:
        mt5 = self._mt5

        # Get current price for market orders
        tick = mt5.symbol_info_tick(order.instrument)
        if not tick:
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                               message=f"No tick data for {order.instrument}")

        price = tick.ask if order.side == Side.BUY else tick.bid
        order_type = (mt5.ORDER_TYPE_BUY if order.side == Side.BUY
                      else mt5.ORDER_TYPE_SELL)

        if order.order_type == OrderType.LIMIT and order.price:
            order_type = (mt5.ORDER_TYPE_BUY_LIMIT if order.side == Side.BUY
                          else mt5.ORDER_TYPE_SELL_LIMIT)
            price = order.price
            action = mt5.TRADE_ACTION_PENDING
        else:
            action = mt5.TRADE_ACTION_DEAL

        request = {
            "action": action,
            "symbol": order.instrument,
            "volume": float(order.quantity),
            "type": order_type,
            "price": float(price),
            "sl": float(order.stop_loss) if order.stop_loss else 0.0,
            "tp": float(order.take_profit) if order.take_profit else 0.0,
            "deviation": 20,
            "magic": 234000,
            "comment": f"atlas:{order.id[:8]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = result.comment if result else str(mt5.last_error())
            log.warning("MT5 order failed", instrument=order.instrument,
                        retcode=result.retcode if result else None, msg=msg)
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED, message=msg)

        return OrderResult(
            order_id=order.id,
            status=OrderStatus.FILLED,
            fill_price=float(result.price),
            fill_quantity=float(order.quantity),
            message=str(result.order),
        )

    async def cancel_order(self, order_id: str) -> bool:
        mt5 = self._mt5
        orders = mt5.orders_get()
        if not orders:
            return False
        for o in orders:
            if str(o.ticket) == order_id:
                result = mt5.order_send({
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": o.ticket,
                })
                return result and result.retcode == mt5.TRADE_RETCODE_DONE
        return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        orders = self._mt5.orders_get()
        if orders:
            for o in orders:
                if str(o.ticket) == order_id:
                    return OrderStatus.SUBMITTED
        return OrderStatus.FILLED

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        orders = self._mt5.orders_get(symbol=instrument) if instrument else self._mt5.orders_get()
        if not orders:
            return []
        result = []
        for o in orders:
            result.append(Order(
                id=str(o.ticket),
                instrument=o.symbol,
                side=Side.BUY if o.type in (0, 2) else Side.SELL,
                order_type=OrderType.MARKET if o.type < 2 else OrderType.LIMIT,
                quantity=float(o.volume_current),
                price=float(o.price_open),
                status=OrderStatus.SUBMITTED,
            ))
        return result

    async def get_positions(self) -> list[Position]:
        positions = self._mt5.positions_get()
        if not positions:
            return []
        result = []
        for p in positions:
            result.append(Position(
                instrument=p.symbol,
                side=Side.BUY if p.type == 0 else Side.SELL,
                quantity=float(p.volume),
                entry_price=float(p.price_open),
                current_price=float(p.price_current),
                unrealized_pnl=float(p.profit),
            ))
        return result

    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        positions = self._mt5.positions_get(symbol=instrument)
        if not positions:
            return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED,
                               message="No open position")
        pos = positions[0]
        qty = quantity or float(pos.volume)
        close_type = (self._mt5.ORDER_TYPE_SELL if pos.type == 0
                      else self._mt5.ORDER_TYPE_BUY)
        tick = self._mt5.symbol_info_tick(instrument)
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": instrument,
            "volume": qty,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "atlas:close",
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        if result and result.retcode == self._mt5.TRADE_RETCODE_DONE:
            return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.FILLED,
                               fill_price=float(result.price), fill_quantity=qty)
        return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED,
                           message=result.comment if result else "Unknown error")

    async def close_all_positions(self) -> list[OrderResult]:
        positions = await self.get_positions()
        results = []
        for pos in positions:
            result = await self.close_position(pos.instrument)
            results.append(result)
        return results
