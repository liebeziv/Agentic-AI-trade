"""Timezone and market hours utilities."""
from __future__ import annotations
from datetime import datetime, time
import pytz


MARKET_HOURS = {
    "forex":     {"start": time(22, 0), "end": time(21, 59), "days": {0,1,2,3,4}},
    "us_stocks": {"start": time(13, 30), "end": time(20, 0),  "days": {0,1,2,3,4}},
    "hk_stocks": {"start": time(1,  30), "end": time(8,  0),  "days": {0,1,2,3,4}},
    "crypto":    {"start": time(0,  0),  "end": time(23,59),  "days": {0,1,2,3,4,5,6}},
    "futures":   {"start": time(23, 0),  "end": time(22, 0),  "days": {0,1,2,3,4}},
}


def utcnow() -> datetime:
    return datetime.now(pytz.UTC).replace(tzinfo=None)


def is_market_open(market: str, dt: datetime | None = None) -> bool:
    dt = dt or utcnow()
    hours = MARKET_HOURS.get(market)
    if not hours:
        return False
    if dt.weekday() not in hours["days"]:
        return False
    current = dt.time().replace(second=0, microsecond=0)
    start, end = hours["start"], hours["end"]
    if start <= end:
        return start <= current <= end
    # Overnight session (e.g. forex: 22:00 - 21:59 next day)
    return current >= start or current <= end


def minutes_to_session_end(market: str, dt: datetime | None = None) -> float:
    """Returns minutes until session closes. Returns 9999 if session not open."""
    dt = dt or utcnow()
    if not is_market_open(market, dt):
        return 9999.0
    hours = MARKET_HOURS.get(market, {})
    end: time = hours.get("end", time(23, 59))
    end_dt = dt.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if end_dt < dt:
        from datetime import timedelta
        end_dt += timedelta(days=1)
    return (end_dt - dt).total_seconds() / 60
