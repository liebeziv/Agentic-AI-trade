"""Abstract exchange adapter interface — unified API for all brokers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from src.types import (
    AccountInfo, Order, OrderResult, OrderStatus, Position, Side,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class ExchangeAdapter(ABC):
    """Unified interface for all exchange/broker connections."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection and authenticate."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnect; cancel pending orders if configured."""

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Return account balance, buying power, margin."""

    @abstractmethod
    async def place_order(self, order: Order) -> OrderResult:
        """Submit order to exchange. Returns fill info or pending status."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll order status by exchange order ID."""

    @abstractmethod
    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        """Return all open/pending orders."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Return all open positions from the exchange."""

    @abstractmethod
    async def close_position(self, instrument: str, quantity: float | None = None) -> OrderResult:
        """Close a position. quantity=None closes the full position."""

    @abstractmethod
    async def close_all_positions(self) -> list[OrderResult]:
        """Emergency: close all positions immediately."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return connection health."""

    @abstractmethod
    def supports_instrument(self, instrument: str) -> bool:
        """Return True if this adapter handles the given instrument."""


class AdapterFactory:
    """Create and cache the right adapter for each instrument type."""

    _adapters: dict[str, ExchangeAdapter] = {}

    @classmethod
    def get_adapter(cls, instrument: str, config: dict) -> ExchangeAdapter:
        from src.data.market_data import _infer_instrument_type

        inst_type = _infer_instrument_type(instrument)

        if inst_type == "us_stocks":
            key = "alpaca"
            if key not in cls._adapters:
                from src.execution.adapters.alpaca_adapter import AlpacaAdapter
                cls._adapters[key] = AlpacaAdapter(config.get("alpaca", {}))
            return cls._adapters[key]

        elif inst_type == "crypto":
            key = "binance"
            if key not in cls._adapters:
                from src.execution.adapters.binance_adapter import BinanceAdapter
                cls._adapters[key] = BinanceAdapter(config.get("binance", {}))
            return cls._adapters[key]

        elif inst_type in ("futures", "hk_stocks"):
            key = "ib"
            if key not in cls._adapters:
                from src.execution.adapters.ib_adapter import IBAdapter
                cls._adapters[key] = IBAdapter(config.get("ib", {}))
            return cls._adapters[key]

        elif inst_type == "forex":
            key = "mt5"
            if key not in cls._adapters:
                from src.execution.adapters.mt5_adapter import MT5Adapter
                cls._adapters[key] = MT5Adapter(config.get("mt5", {}))
            return cls._adapters[key]

        raise ValueError(f"No adapter available for instrument: {instrument}")

    @classmethod
    async def connect_all(cls) -> None:
        for name, adapter in cls._adapters.items():
            try:
                await adapter.connect()
                log.info("Adapter connected", name=name)
            except Exception as exc:
                log.error("Adapter connect failed", name=name, error=str(exc))

    @classmethod
    async def disconnect_all(cls) -> None:
        for name, adapter in cls._adapters.items():
            try:
                await adapter.disconnect()
            except Exception:
                pass
        cls._adapters.clear()

    @classmethod
    async def close_all_positions(cls) -> None:
        for name, adapter in cls._adapters.items():
            try:
                await adapter.close_all_positions()
                log.info("All positions closed", adapter=name)
            except Exception as exc:
                log.error("Close all failed", adapter=name, error=str(exc))
