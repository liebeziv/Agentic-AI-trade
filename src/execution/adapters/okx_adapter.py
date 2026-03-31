"""OKX adapter — spot and derivatives trading via ccxt."""
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


def _map_okx_status(status: str) -> OrderStatus:
    mapping: dict[str, OrderStatus] = {
        "live":             OrderStatus.SUBMITTED,
        "partially_filled": OrderStatus.PARTIAL_FILL,
        "filled":           OrderStatus.FILLED,
        "canceled":         OrderStatus.CANCELLED,
        "mmp_canceled":     OrderStatus.CANCELLED,
    }
    return mapping.get(status.lower(), OrderStatus.PENDING)


def _side_to_okx(side: Side) -> str:
    return "buy" if side == Side.BUY else "sell"


def _okx_to_side(side_str: str) -> Side:
    return Side.BUY if side_str.lower() == "buy" else Side.SELL


def _okx_to_order_type(type_str: str) -> OrderType:
    return OrderType.LIMIT if type_str.lower() == "limit" else OrderType.MARKET


class OKXAdapter(ExchangeAdapter):
    """OKX exchange adapter supporting spot and USDT-margined derivatives via ccxt."""

    def __init__(self, config: dict) -> None:
        self.api_key: str = (
            config.get("api_key") or os.getenv("OKX_API_KEY", "")
        )
        self.secret: str = (
            config.get("secret") or os.getenv("OKX_SECRET", "")
        )
        self.passphrase: str = (
            config.get("passphrase") or os.getenv("OKX_PASSPHRASE", "")
        )
        self.sandbox: bool = config.get("sandbox", True)
        self._exchange = None
        self._connected = False

    async def connect(self) -> None:
        """Create ccxt.okx instance and verify connectivity."""
        try:
            import ccxt.async_support as ccxt  # type: ignore
        except ImportError:
            raise ImportError("ccxt not installed: pip install ccxt")

        if not self.api_key or not self.secret or not self.passphrase:
            raise ValueError(
                "OKX_API_KEY, OKX_SECRET, and OKX_PASSPHRASE must all be set"
            )

        self._exchange = ccxt.okx(
            {
                "apiKey": self.api_key,
                "secret": self.secret,
                "password": self.passphrase,
                "enableRateLimit": True,
            }
        )

        if self.sandbox:
            self._exchange.set_sandbox_mode(True)

        try:
            # Lightweight connectivity test
            await self._exchange.load_markets()
            self._connected = True
            log.info("OKX connected", sandbox=self.sandbox)
        except Exception as exc:
            await self._exchange.close()
            self._exchange = None
            raise ConnectionError(f"OKX connect failed: {exc}") from exc

    async def disconnect(self) -> None:
        self._connected = False
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception as exc:
                log.warning("OKX disconnect error", error=str(exc))
            self._exchange = None

    def is_connected(self) -> bool:
        return self._connected and self._exchange is not None

    def supports_instrument(self, instrument: str) -> bool:
        """OKX supports crypto spot (BTC/USDT) and derivatives (BTC-USDT-SWAP)."""
        return "/" in instrument or instrument.endswith("-SWAP") or instrument.endswith("-PERP")

    async def get_account(self) -> AccountInfo:
        try:
            balance = await self._exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            total_equity = float(usdt.get("total", 0.0) or 0.0)
            free_cash = float(usdt.get("free", 0.0) or 0.0)
            used_margin = float(usdt.get("used", 0.0) or 0.0)
            return AccountInfo(
                equity=total_equity,
                cash=free_cash,
                buying_power=free_cash,
                margin_used=used_margin,
                currency="USDT",
            )
        except Exception as exc:
            log.warning("OKX get_account failed", error=str(exc))
            return AccountInfo(equity=0.0, cash=0.0, buying_power=0.0)

    async def place_order(self, order: Order) -> OrderResult:
        try:
            side = _side_to_okx(order.side)
            symbol = order.instrument

            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=order.quantity,
                )
            elif order.order_type == OrderType.LIMIT and order.price is not None:
                result = await self._exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side=side,
                    amount=order.quantity,
                    price=order.price,
                )
            else:
                return OrderResult(
                    order_id=order.id,
                    status=OrderStatus.REJECTED,
                    message="Unsupported order type or missing limit price",
                )

            raw_status = result.get("status", "") or result.get("info", {}).get("state", "")
            status = _map_okx_status(raw_status)

            avg_price = result.get("average")
            fill_price = float(avg_price) if avg_price else None
            filled_qty = float(result.get("filled") or 0.0)
            exchange_id: str = str(result.get("id", ""))

            return OrderResult(
                order_id=order.id,
                status=status,
                fill_price=fill_price,
                fill_quantity=filled_qty,
                message=exchange_id,
            )
        except Exception as exc:
            log.warning("OKX place_order failed", instrument=order.instrument, error=str(exc))
            return OrderResult(
                order_id=order.id,
                status=OrderStatus.REJECTED,
                message=str(exc),
            )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception as exc:
            log.warning("OKX cancel_order failed", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            order = await self._exchange.fetch_order(order_id)
            raw_status = order.get("status", "") or order.get("info", {}).get("state", "")
            return _map_okx_status(raw_status)
        except Exception as exc:
            log.warning("OKX get_order_status failed", order_id=order_id, error=str(exc))
            return OrderStatus.REJECTED

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        try:
            raw_orders = await self._exchange.fetch_open_orders(symbol=instrument)
            result: list[Order] = []
            for o in raw_orders:
                raw_status = o.get("status", "") or o.get("info", {}).get("state", "")
                price_val = o.get("price")
                result.append(Order(
                    id=str(o.get("clientOrderId") or o.get("id", "")),
                    instrument=o.get("symbol", instrument or ""),
                    side=_okx_to_side(o.get("side", "buy")),
                    order_type=_okx_to_order_type(o.get("type", "market")),
                    quantity=float(o.get("amount") or 0.0),
                    price=float(price_val) if price_val else None,
                    status=_map_okx_status(raw_status),
                    created_at=datetime.utcfromtimestamp(
                        o["timestamp"] / 1000
                    ) if o.get("timestamp") else datetime.utcnow(),
                ))
            return result
        except Exception as exc:
            log.warning("OKX get_open_orders failed", instrument=instrument, error=str(exc))
            return []

    async def get_positions(self) -> list[Position]:
        try:
            raw_positions = await self._exchange.fetch_positions()
            result: list[Position] = []
            for p in raw_positions:
                contracts = float(p.get("contracts") or p.get("info", {}).get("pos", 0) or 0.0)
                if contracts == 0.0:
                    continue
                side_str = p.get("side", "long")
                side = Side.BUY if side_str.lower() in ("long", "buy") else Side.SELL
                entry = float(p.get("entryPrice") or 0.0)
                mark = float(p.get("markPrice") or p.get("info", {}).get("markPx", 0.0) or 0.0)
                upnl = float(p.get("unrealizedPnl") or 0.0)
                result.append(Position(
                    instrument=p.get("symbol", ""),
                    side=side,
                    quantity=abs(contracts),
                    entry_price=entry,
                    current_price=mark,
                    unrealized_pnl=upnl,
                ))
            return result
        except Exception as exc:
            log.warning("OKX get_positions failed", error=str(exc))
            return []

    async def get_current_price(self, instrument: str) -> float:
        """Fetch the latest mid price for an instrument via the ticker endpoint."""
        try:
            ticker = await self._exchange.fetch_ticker(instrument)
            last = ticker.get("last")
            if last is not None:
                return float(last)
            bid = ticker.get("bid") or 0.0
            ask = ticker.get("ask") or 0.0
            if bid and ask:
                return (float(bid) + float(ask)) / 2.0
            return float(bid or ask or 0.0)
        except Exception as exc:
            log.warning("OKX get_current_price failed", instrument=instrument, error=str(exc))
            return 0.0

    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        try:
            positions = await self.get_positions()
            target = next((p for p in positions if p.instrument == instrument), None)
            if target is None:
                return OrderResult(
                    order_id=f"CLOSE-{instrument}",
                    status=OrderStatus.REJECTED,
                    message=f"No open position for {instrument}",
                )

            close_qty = quantity if quantity is not None else target.quantity
            close_side = Side.SELL if target.side == Side.BUY else Side.BUY

            close_order = Order(
                id=f"CLOSE-{instrument}",
                instrument=instrument,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=close_qty,
            )
            return await self.place_order(close_order)
        except Exception as exc:
            log.warning("OKX close_position failed", instrument=instrument, error=str(exc))
            return OrderResult(
                order_id=f"CLOSE-{instrument}",
                status=OrderStatus.REJECTED,
                message=str(exc),
            )

    async def close_all_positions(self) -> list[OrderResult]:
        try:
            positions = await self.get_positions()
            if not positions:
                log.info("OKX: no open positions to close")
                return []

            results: list[OrderResult] = []
            for pos in positions:
                result = await self.close_position(pos.instrument, pos.quantity)
                results.append(result)
                log.info(
                    "OKX: close position submitted",
                    instrument=pos.instrument,
                    qty=pos.quantity,
                    status=result.status,
                )
            return results
        except Exception as exc:
            log.error("OKX close_all_positions failed", error=str(exc))
            return [OrderResult(order_id="CLOSE_ALL", status=OrderStatus.REJECTED, message=str(exc))]
