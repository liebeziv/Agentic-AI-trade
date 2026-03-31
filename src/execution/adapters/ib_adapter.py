"""Interactive Brokers adapter for futures and HK stocks via ib_insync."""
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

FUTURES_CONTRACTS: dict[str, tuple[str, str, str]] = {
    "ES": ("ES", "CME", "USD"),
    "NQ": ("NQ", "CME", "USD"),
    "GC": ("GC", "COMEX", "USD"),
    "CL": ("CL", "NYMEX", "USD"),
    "SI": ("SI", "COMEX", "USD"),
    "ZB": ("ZB", "CBOT", "USD"),
}


class IBAdapter(ExchangeAdapter):
    """IB adapter (requires TWS/Gateway running on localhost)."""

    def __init__(self, config: dict) -> None:
        self.host = config.get("host") or os.getenv("IB_HOST", "127.0.0.1")
        self.port = int(config.get("port") or os.getenv("IB_PORT", 7497))
        self.client_id = int(config.get("client_id") or os.getenv("IB_CLIENT_ID", 1))
        self._ib = None
        self._connected = False

    async def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError("ib_insync not installed: pip install ib_insync")

        self._ib = IB()
        await self._ib.connectAsync(self.host, self.port, clientId=self.client_id)
        self._connected = True
        log.info("IB connected", host=self.host, port=self.port,
                 paper=(self.port == 7497))

    async def disconnect(self) -> None:
        if self._ib:
            self._ib.disconnect()
        self._connected = False
        self._ib = None

    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    def supports_instrument(self, instrument: str) -> bool:
        return instrument in FUTURES_CONTRACTS or instrument.endswith(".HK")

    async def get_account(self) -> AccountInfo:
        summary = await self._ib.accountSummaryAsync()
        vals: dict[str, float] = {}
        for item in summary:
            try:
                vals[item.tag] = float(item.value)
            except (ValueError, AttributeError):
                pass
        return AccountInfo(
            equity=vals.get("NetLiquidation", 0.0),
            cash=vals.get("CashBalance", 0.0),
            buying_power=vals.get("BuyingPower", 0.0),
            margin_used=vals.get("MaintMarginReq", 0.0),
            currency="USD",
        )

    def _resolve_contract(self, instrument: str):
        from ib_insync import Future, Stock
        if instrument in FUTURES_CONTRACTS:
            sym, exchange, currency = FUTURES_CONTRACTS[instrument]
            return Future(sym, exchange=exchange, currency=currency)
        if instrument.endswith(".HK"):
            sym = instrument.replace(".HK", "")
            return Stock(sym, "SEHK", "HKD")
        return Stock(instrument, "SMART", "USD")

    async def place_order(self, order: Order) -> OrderResult:
        from ib_insync import MarketOrder, LimitOrder

        contract = self._resolve_contract(order.instrument)
        action = "BUY" if order.side == Side.BUY else "SELL"

        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT and order.price:
            ib_order = LimitOrder(action, order.quantity, order.price)
        else:
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                               message="Unsupported order type")

        try:
            trade = self._ib.placeOrder(contract, ib_order)
            # Brief wait for acknowledgement
            await asyncio.sleep(2)

            status_str = str(trade.orderStatus.status).lower()
            fill_price = trade.orderStatus.avgFillPrice or None
            fill_qty = trade.orderStatus.filled or 0.0
            commission = sum(f.commissionReport.commission for f in trade.fills
                             if f.commissionReport) if trade.fills else 0.0

            return OrderResult(
                order_id=order.id,
                status=OrderStatus.FILLED if "filled" in status_str else OrderStatus.SUBMITTED,
                fill_price=fill_price if fill_price else None,
                fill_quantity=float(fill_qty),
                commission=float(commission),
                message=str(trade.order.orderId),
            )
        except Exception as exc:
            log.warning("IB order failed", instrument=order.instrument, error=str(exc))
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED, message=str(exc))

    async def cancel_order(self, order_id: str) -> bool:
        try:
            open_trades = self._ib.openTrades()
            for trade in open_trades:
                if str(trade.order.orderId) == order_id:
                    self._ib.cancelOrder(trade.order)
                    return True
            return False
        except Exception as exc:
            log.warning("IB cancel failed", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        open_trades = self._ib.openTrades()
        for trade in open_trades:
            if str(trade.order.orderId) == order_id:
                status_str = str(trade.orderStatus.status).lower()
                if "filled" in status_str:
                    return OrderStatus.FILLED
                if "cancel" in status_str:
                    return OrderStatus.CANCELLED
                return OrderStatus.SUBMITTED
        return OrderStatus.FILLED  # Not in open trades = likely filled

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        open_trades = self._ib.openTrades()
        result = []
        for trade in open_trades:
            sym = trade.contract.symbol
            if instrument and sym != instrument:
                continue
            result.append(Order(
                id=str(trade.order.orderId),
                instrument=sym,
                side=Side.BUY if trade.order.action == "BUY" else Side.SELL,
                order_type=OrderType.MARKET if trade.order.orderType == "MKT" else OrderType.LIMIT,
                quantity=float(trade.order.totalQuantity),
                price=float(trade.order.lmtPrice) if trade.order.lmtPrice else None,
                status=OrderStatus.SUBMITTED,
            ))
        return result

    async def get_positions(self) -> list[Position]:
        portfolio = self._ib.portfolio()
        result = []
        for item in portfolio:
            qty = float(item.position)
            if qty == 0:
                continue
            result.append(Position(
                instrument=item.contract.symbol,
                side=Side.BUY if qty > 0 else Side.SELL,
                quantity=abs(qty),
                entry_price=float(item.averageCost),
                current_price=float(item.marketPrice) if item.marketPrice else 0.0,
                unrealized_pnl=float(item.unrealizedPNL) if item.unrealizedPNL else 0.0,
            ))
        return result

    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        from ib_insync import MarketOrder

        positions = await self.get_positions()
        pos = next((p for p in positions if p.instrument == instrument), None)
        if not pos:
            return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED,
                               message="No open position")
        qty = quantity or pos.quantity
        action = "SELL" if pos.side == Side.BUY else "BUY"
        contract = self._resolve_contract(instrument)
        trade = self._ib.placeOrder(contract, MarketOrder(action, qty))
        await asyncio.sleep(1)
        return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.SUBMITTED,
                           fill_quantity=qty)

    async def close_all_positions(self) -> list[OrderResult]:
        positions = await self.get_positions()
        results = []
        for pos in positions:
            result = await self.close_position(pos.instrument)
            results.append(result)
        return results
