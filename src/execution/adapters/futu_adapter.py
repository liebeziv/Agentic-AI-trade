"""Futu OpenD adapter — HK stock trading via futu-api."""
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

# ---------------------------------------------------------------------------
# Futu order-status mapping
# ---------------------------------------------------------------------------
_FUTU_STATUS_MAP: dict[str, OrderStatus] = {
    "SUBMITTED":    OrderStatus.SUBMITTED,
    "FILLED_ALL":   OrderStatus.FILLED,
    "FILLED_PART":  OrderStatus.PARTIAL_FILL,
    "CANCELLED_ALL": OrderStatus.CANCELLED,
    "CANCELLED_PART": OrderStatus.CANCELLED,
    "FAILED":       OrderStatus.REJECTED,
    "DISABLED":     OrderStatus.REJECTED,
    "DELETED":      OrderStatus.CANCELLED,
    "WAITING_SUBMIT": OrderStatus.PENDING,
    "SUBMITTING":   OrderStatus.SUBMITTED,
}


def _map_futu_status(status: str) -> OrderStatus:
    return _FUTU_STATUS_MAP.get(status.upper(), OrderStatus.PENDING)


def _side_to_futu(side: Side):
    """Return futu TrdSide enum value."""
    import futu  # imported lazily to stay inside guard
    return futu.TrdSide.BUY if side == Side.BUY else futu.TrdSide.SELL


def _futu_to_side(trd_side_val) -> Side:
    """Convert futu TrdSide int/str to our Side enum."""
    try:
        import futu
        if trd_side_val == futu.TrdSide.BUY:
            return Side.BUY
    except Exception:
        pass
    # Fallback: string comparison
    s = str(trd_side_val).upper()
    return Side.BUY if "BUY" in s else Side.SELL


def _order_type_to_futu(order_type: OrderType):
    """Return futu OrderType enum value."""
    import futu
    if order_type == OrderType.MARKET:
        return futu.OrderType.MARKET
    return futu.OrderType.NORMAL  # NORMAL == limit order in Futu


def _futu_to_order_type(futu_type_val) -> OrderType:
    try:
        import futu
        if futu_type_val == futu.OrderType.MARKET:
            return OrderType.MARKET
    except Exception:
        pass
    return OrderType.LIMIT


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FutuAdapter(ExchangeAdapter):
    """Futu OpenD adapter for HK stock trading.

    Requires a running Futu OpenD gateway process.  Connect credentials are
    read from *config* dict or environment variables:

    - FUTU_HOST          — OpenD host (default "127.0.0.1")
    - FUTU_PORT          — OpenD port (default 11111)
    - FUTU_TRADE_PWD     — Trade-unlock password (REAL account only)
    - FUTU_ACCOUNT_TYPE  — "SIMULATE" (default) or "REAL"
    """

    def __init__(self, config: dict) -> None:
        self.host: str = (
            config.get("host") or os.getenv("FUTU_HOST", "127.0.0.1")
        )
        self.port: int = int(
            config.get("port") or os.getenv("FUTU_PORT", "11111")
        )
        self.trade_pwd: str = (
            config.get("trade_pwd") or os.getenv("FUTU_TRADE_PWD", "")
        )
        # "SIMULATE" maps to futu TrdEnv.SIMULATE; "REAL" → TrdEnv.REAL
        raw_env: str = (
            config.get("account_type") or os.getenv("FUTU_ACCOUNT_TYPE", "SIMULATE")
        ).upper()
        self._is_real: bool = raw_env == "REAL"

        self._trd_ctx = None   # futu.OpenHKTradeContext
        self._quote_ctx = None  # futu.OpenQuoteContext
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open Futu trade and quote contexts; unlock trade if REAL account."""
        try:
            import futu  # type: ignore
        except ImportError:
            raise ImportError(
                "futu-api is not installed. Install it with: pip install futu-api"
            )

        try:
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE

            self._quote_ctx = futu.OpenQuoteContext(host=self.host, port=self.port)
            self._trd_ctx = futu.OpenHKTradeContext(host=self.host, port=self.port)

            # Unlock trade password for real accounts
            if self._is_real:
                if not self.trade_pwd:
                    raise ValueError(
                        "FUTU_TRADE_PWD must be set for REAL account trading"
                    )
                ret, data = self._trd_ctx.unlock_trade(self.trade_pwd)
                if ret != futu.RET_OK:
                    raise ConnectionError(f"Futu trade unlock failed: {data}")

            self._connected = True
            log.info(
                "Futu connected",
                host=self.host,
                port=self.port,
                env="REAL" if self._is_real else "SIMULATE",
            )
        except Exception as exc:
            self._connected = False
            if self._trd_ctx is not None:
                try:
                    self._trd_ctx.close()
                except Exception:
                    pass
                self._trd_ctx = None
            if self._quote_ctx is not None:
                try:
                    self._quote_ctx.close()
                except Exception:
                    pass
                self._quote_ctx = None
            raise ConnectionError(f"Futu connect failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Close both Futu contexts."""
        self._connected = False
        for ctx_attr in ("_trd_ctx", "_quote_ctx"):
            ctx = getattr(self, ctx_attr, None)
            if ctx is not None:
                try:
                    ctx.close()
                except Exception as exc:
                    log.warning("Futu context close error", ctx=ctx_attr, error=str(exc))
                setattr(self, ctx_attr, None)
        log.info("Futu disconnected")

    def is_connected(self) -> bool:
        return self._connected and self._trd_ctx is not None

    def supports_instrument(self, instrument: str) -> bool:
        """Return True for HK-listed stocks (e.g. '00700.HK' or market=='HK')."""
        upper = instrument.upper()
        return upper.endswith(".HK") or upper.startswith("HK.")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    async def get_account(self) -> AccountInfo:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE
            ret, data = self._trd_ctx.accinfo_query(trd_env=trd_env)
            if ret != futu.RET_OK:
                log.warning("Futu accinfo_query failed", detail=str(data))
                return AccountInfo(equity=0.0, cash=0.0, buying_power=0.0)

            row = data.iloc[0] if hasattr(data, "iloc") else data[0]
            equity = float(row.get("total_assets", 0.0) or 0.0)
            cash = float(row.get("cash", 0.0) or 0.0)
            buying_power = float(row.get("max_power_short", cash) or cash)
            return AccountInfo(
                equity=equity,
                cash=cash,
                buying_power=buying_power,
                currency="HKD",
            )
        except Exception as exc:
            log.warning("Futu get_account failed", error=str(exc))
            return AccountInfo(equity=0.0, cash=0.0, buying_power=0.0)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> OrderResult:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE
            trd_side = _side_to_futu(order.side)
            futu_order_type = _order_type_to_futu(order.order_type)

            # Futu place_order requires price even for market orders;
            # pass 0 for market and let OpenD handle it.
            price = order.price if order.price is not None else 0.0

            ret, data = self._trd_ctx.place_order(
                price=price,
                qty=order.quantity,
                code=order.instrument,
                trd_side=trd_side,
                order_type=futu_order_type,
                trd_env=trd_env,
            )

            if ret != futu.RET_OK:
                log.warning(
                    "Futu place_order failed",
                    instrument=order.instrument,
                    detail=str(data),
                )
                return OrderResult(
                    order_id=order.id,
                    status=OrderStatus.REJECTED,
                    message=str(data),
                )

            row = data.iloc[0] if hasattr(data, "iloc") else data[0]
            exchange_order_id = str(row.get("order_id", ""))
            raw_status = str(row.get("order_status", "SUBMITTED"))
            status = _map_futu_status(raw_status)

            fill_price_raw = row.get("dealt_avg_price")
            fill_price = float(fill_price_raw) if fill_price_raw else None
            fill_qty = float(row.get("dealt_qty", 0.0) or 0.0)

            return OrderResult(
                order_id=order.id,
                status=status,
                fill_price=fill_price,
                fill_quantity=fill_qty,
                message=exchange_order_id,
            )
        except Exception as exc:
            log.warning("Futu place_order exception", instrument=order.instrument, error=str(exc))
            return OrderResult(
                order_id=order.id,
                status=OrderStatus.REJECTED,
                message=str(exc),
            )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE
            ret, data = self._trd_ctx.modify_order(
                modify_order_op=futu.ModifyOrderOp.CANCEL,
                order_id=order_id,
                qty=0,
                price=0,
                trd_env=trd_env,
            )
            if ret != futu.RET_OK:
                log.warning("Futu cancel_order failed", order_id=order_id, detail=str(data))
                return False
            return True
        except Exception as exc:
            log.warning("Futu cancel_order exception", order_id=order_id, error=str(exc))
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE
            ret, data = self._trd_ctx.order_list_query(
                order_id=order_id,
                trd_env=trd_env,
            )
            if ret != futu.RET_OK:
                log.warning("Futu order_list_query failed", order_id=order_id, detail=str(data))
                return OrderStatus.REJECTED

            if hasattr(data, "empty") and data.empty:
                return OrderStatus.REJECTED
            if hasattr(data, "__len__") and len(data) == 0:
                return OrderStatus.REJECTED

            row = data.iloc[0] if hasattr(data, "iloc") else data[0]
            raw_status = str(row.get("order_status", ""))
            return _map_futu_status(raw_status)
        except Exception as exc:
            log.warning("Futu get_order_status exception", order_id=order_id, error=str(exc))
            return OrderStatus.REJECTED

    async def get_open_orders(self, instrument: str | None = None) -> list[Order]:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE

            kwargs: dict = {"trd_env": trd_env}
            if instrument:
                kwargs["code"] = instrument

            # Query only active/open statuses
            ret, data = self._trd_ctx.order_list_query(
                status_filter_list=[
                    futu.OrderStatus.WAITING_SUBMIT,
                    futu.OrderStatus.SUBMITTING,
                    futu.OrderStatus.SUBMITTED,
                    futu.OrderStatus.FILLED_PART,
                ],
                **kwargs,
            )
            if ret != futu.RET_OK:
                log.warning("Futu get_open_orders failed", detail=str(data))
                return []

            rows = list(data.itertuples(index=False)) if hasattr(data, "itertuples") else data
            result: list[Order] = []
            for row in rows:
                _get = (lambda r, k, d=None: getattr(r, k, None) or d)
                raw_status = str(_get(row, "order_status", "SUBMITTED"))
                price_val = _get(row, "price")
                create_time_str = _get(row, "create_time", "")
                try:
                    created_at = datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    created_at = datetime.utcnow()

                result.append(Order(
                    id=str(_get(row, "order_id", "")),
                    instrument=str(_get(row, "code", instrument or "")),
                    side=_futu_to_side(_get(row, "trd_side")),
                    order_type=_futu_to_order_type(_get(row, "order_type")),
                    quantity=float(_get(row, "qty", 0.0) or 0.0),
                    price=float(price_val) if price_val else None,
                    status=_map_futu_status(raw_status),
                    created_at=created_at,
                ))
            return result
        except Exception as exc:
            log.warning("Futu get_open_orders exception", instrument=instrument, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        try:
            import futu
            trd_env = futu.TrdEnv.REAL if self._is_real else futu.TrdEnv.SIMULATE
            ret, data = self._trd_ctx.position_list_query(trd_env=trd_env)
            if ret != futu.RET_OK:
                log.warning("Futu position_list_query failed", detail=str(data))
                return []

            rows = list(data.itertuples(index=False)) if hasattr(data, "itertuples") else data
            result: list[Position] = []
            for row in rows:
                _get = (lambda r, k, d=None: getattr(r, k, None) or d)
                qty = float(_get(row, "qty", 0.0) or 0.0)
                if qty == 0.0:
                    continue
                entry = float(_get(row, "cost_price", 0.0) or 0.0)
                current = float(_get(row, "nominal_price", 0.0) or 0.0)
                upnl = float(_get(row, "unrealized_pl", 0.0) or 0.0)
                rpnl = float(_get(row, "realized_pl", 0.0) or 0.0)

                result.append(Position(
                    instrument=str(_get(row, "code", "")),
                    side=Side.BUY,   # Futu HK positions are always long (no short for retail)
                    quantity=qty,
                    entry_price=entry,
                    current_price=current,
                    unrealized_pnl=upnl,
                    realized_pnl=rpnl,
                ))
            return result
        except Exception as exc:
            log.warning("Futu get_positions exception", error=str(exc))
            return []

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
            log.warning("Futu close_position failed", instrument=instrument, error=str(exc))
            return OrderResult(
                order_id=f"CLOSE-{instrument}",
                status=OrderStatus.REJECTED,
                message=str(exc),
            )

    async def close_all_positions(self) -> list[OrderResult]:
        try:
            positions = await self.get_positions()
            if not positions:
                log.info("Futu: no open positions to close")
                return []

            results: list[OrderResult] = []
            for pos in positions:
                result = await self.close_position(pos.instrument, pos.quantity)
                results.append(result)
                log.info(
                    "Futu: close position submitted",
                    instrument=pos.instrument,
                    qty=pos.quantity,
                    status=result.status,
                )
            return results
        except Exception as exc:
            log.error("Futu close_all_positions failed", error=str(exc))
            return [OrderResult(order_id="CLOSE_ALL", status=OrderStatus.REJECTED, message=str(exc))]

    # ------------------------------------------------------------------
    # Optional extra: current quote price via OpenQuoteContext
    # ------------------------------------------------------------------

    async def get_current_price(self, instrument: str) -> float:
        """Fetch the latest ask/bid mid price for a HK instrument."""
        try:
            import futu
            ret, data = self._quote_ctx.get_stock_quote([instrument])
            if ret != futu.RET_OK:
                log.warning("Futu get_current_price failed", instrument=instrument, detail=str(data))
                return 0.0
            row = data.iloc[0] if hasattr(data, "iloc") else data[0]
            _get = lambda k, d=0.0: float(getattr(row, k, None) or d)
            last = _get("last_price")
            if last:
                return last
            bid = _get("bid_price")
            ask = _get("ask_price")
            if bid and ask:
                return (bid + ask) / 2.0
            return float(bid or ask or 0.0)
        except Exception as exc:
            log.warning("Futu get_current_price exception", instrument=instrument, error=str(exc))
            return 0.0
