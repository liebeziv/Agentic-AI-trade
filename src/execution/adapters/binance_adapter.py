"""Binance adapter for crypto spot trading via ccxt (testnet by default)."""
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


def _map_ccxt_status(status: str) -> OrderStatus:
    mapping = {
        "open": OrderStatus.SUBMITTED,
        "closed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "rejected": OrderStatus.REJECTED,
        "partially_filled": OrderStatus.PARTIAL_FILL,
    }
    return mapping.get(status, OrderStatus.PENDING)


class BinanceAdapter(ExchangeAdapter):
    """Binance spot trading via ccxt.async_support (testnet by default)."""

    def __init__(self, config: dict) -> None:
        self.api_key = config.get("api_key") or os.getenv("BINANCE_API_KEY", "")
        self.secret = config.get("secret") or os.getenv("BINANCE_SECRET", "")
        self.testnet = config.get("testnet", True)
        self._exchange = None
        self._connected = False

    async def connect(self) -> None:
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            raise ImportError("ccxt not installed: pip install ccxt")

        options: dict = {"defaultType": "spot"}
        if self.testnet:
            options["urls"] = {
                "api": {
                    "public": "https://testnet.binance.vision/api",
                    "private": "https://testnet.binance.vision/api",
                }
            }

        self._exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.secret,
            "options": options,
            "enableRateLimit": True,
        })

        if not self.api_key:
            # No key — public-only mode (read market data, no trading)
            log.warning("Binance: no API key — read-only mode")
            self._connected = True
            return

        await self._exchange.load_markets()
        balance = await self._exchange.fetch_balance()
        self._connected = True
        log.info("Binance connected", testnet=self.testnet,
                 usdt_balance=balance.get("USDT", {}).get("free", 0))

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()
        self._connected = False
        self._exchange = None

    def is_connected(self) -> bool:
        return self._connected and self._exchange is not None

    def supports_instrument(self, instrument: str) -> bool:
        return "/" in instrument  # crypto pairs like BTC/USDT

    async def get_account(self) -> AccountInfo:
        balance = await self._exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        total = sum(v.get("total", 0) * 1.0 for v in balance.values()
                    if isinstance(v, dict) and "total" in v)
        return AccountInfo(
            equity=float(total),
            cash=float(usdt.get("free", 0)),
            buying_power=float(usdt.get("free", 0)),
        )

    async def place_order(self, order: Order) -> OrderResult:
        if not self.api_key:
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                               message="No API key configured")
        side = "buy" if order.side == Side.BUY else "sell"
        try:
            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_market_order(
                    order.instrument, side, order.quantity
                )
            elif order.order_type == OrderType.LIMIT and order.price:
                result = await self._exchange.create_limit_order(
                    order.instrument, side, order.quantity, order.price
                )
            else:
                return OrderResult(order_id=order.id, status=OrderStatus.REJECTED,
                                   message="Unsupported order type")

            fee = 0.0
            if result.get("fee"):
                fee = float(result["fee"].get("cost", 0))

            return OrderResult(
                order_id=order.id,
                status=_map_ccxt_status(result.get("status", "open")),
                fill_price=result.get("average") or result.get("price"),
                fill_quantity=float(result.get("filled", 0)),
                commission=fee,
                message=str(result.get("id", "")),
            )
        except Exception as exc:
            log.warning("Binance order failed", instrument=order.instrument, error=str(exc))
            return OrderResult(order_id=order.id, status=OrderStatus.REJECTED, message=str(exc))

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception as exc:
            log.warning("Binance cancel failed", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            order = await self._exchange.fetch_order(order_id)
            return _map_ccxt_status(order.get("status", "open"))
        except Exception:
            return OrderStatus.REJECTED

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        try:
            raw = await self._exchange.fetch_open_orders(instrument)
        except Exception:
            return []
        result = []
        for o in raw:
            result.append(Order(
                id=str(o.get("clientOrderId") or o["id"]),
                instrument=o["symbol"],
                side=Side.BUY if o["side"] == "buy" else Side.SELL,
                order_type=OrderType.MARKET if o["type"] == "market" else OrderType.LIMIT,
                quantity=float(o["amount"]),
                price=float(o["price"]) if o.get("price") else None,
                status=_map_ccxt_status(o.get("status", "open")),
            ))
        return result

    async def get_positions(self) -> list[Position]:
        """For spot trading, positions = non-zero balances."""
        try:
            balance = await self._exchange.fetch_balance()
        except Exception:
            return []

        positions = []
        for asset, info in balance.items():
            if not isinstance(info, dict):
                continue
            total = float(info.get("total", 0))
            if total <= 0 or asset in ("USDT", "USD", "info", "free", "used", "total"):
                continue
            # Try to get current price
            try:
                symbol = f"{asset}/USDT"
                ticker = await self._exchange.fetch_ticker(symbol)
                price = float(ticker.get("last", 0))
            except Exception:
                price = 0.0

            positions.append(Position(
                instrument=f"{asset}/USDT",
                side=Side.BUY,
                quantity=total,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0,
            ))
        return positions

    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        try:
            ticker = await self._exchange.fetch_ticker(instrument)
            price = float(ticker["last"])
            balance = await self._exchange.fetch_balance()
            asset = instrument.split("/")[0]
            qty = quantity or float(balance.get(asset, {}).get("free", 0))
            if qty <= 0:
                return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED,
                                   message="No position to close")
            result = await self._exchange.create_market_order(instrument, "sell", qty)
            return OrderResult(
                order_id=f"CLOSE-{instrument}",
                status=_map_ccxt_status(result.get("status", "open")),
                fill_price=result.get("average"),
                fill_quantity=float(result.get("filled", 0)),
            )
        except Exception as exc:
            return OrderResult(order_id=f"CLOSE-{instrument}", status=OrderStatus.REJECTED,
                               message=str(exc))

    async def close_all_positions(self) -> list[OrderResult]:
        positions = await self.get_positions()
        results = []
        for pos in positions:
            result = await self.close_position(pos.instrument, pos.quantity)
            results.append(result)
        return results
