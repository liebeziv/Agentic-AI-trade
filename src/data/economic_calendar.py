"""Economic calendar — ForexFactory weekly JSON feed."""
from __future__ import annotations
from datetime import datetime, timedelta

import aiohttp

from src.types import EconomicEvent
from src.utils.logger import get_logger

log = get_logger(__name__)

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class EconomicCalendar:
    def __init__(self) -> None:
        self._events: list[EconomicEvent] = []
        self._last_fetch: datetime | None = None

    async def refresh(self) -> None:
        """Fetch this week's economic calendar from ForexFactory."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(FF_CALENDAR_URL, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        log.warning("ForexFactory calendar returned non-200", status=resp.status)
                        return
                    data = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("Economic calendar fetch failed", error=str(exc))
            return

        events: list[EconomicEvent] = []
        for item in data:
            try:
                ts_str = item.get("date", "") + " " + item.get("time", "00:00am")
                # Parse "03-31-2026 8:30am" style
                ts = datetime.strptime(ts_str.strip(), "%m-%d-%Y %I:%M%p")
            except Exception:
                try:
                    ts = datetime.fromisoformat(item.get("date", "2000-01-01"))
                except Exception:
                    continue

            impact_map = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW"}
            events.append(EconomicEvent(
                timestamp=ts,
                name=item.get("title", ""),
                country=item.get("country", ""),
                impact=impact_map.get(item.get("impact", "Low"), "LOW"),
                forecast=str(item.get("forecast", "")),
                previous=str(item.get("previous", "")),
                actual=str(item.get("actual", "")),
            ))

        self._events = events
        self._last_fetch = datetime.utcnow()
        log.info("Economic calendar refreshed", count=len(events))

    def upcoming_events(
        self, hours: int = 4, impact_filter: list[str] | None = None
    ) -> list[EconomicEvent]:
        """Return events in the next N hours, optionally filtered by impact."""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours)
        result = [e for e in self._events if now <= e.timestamp <= end]
        if impact_filter:
            result = [e for e in result if e.impact in impact_filter]
        return result

    def has_high_impact_soon(self, minutes: int = 15) -> bool:
        """True if a HIGH-impact event is within `minutes` minutes."""
        threshold = datetime.utcnow() + timedelta(minutes=minutes)
        return any(
            e.impact == "HIGH" and e.timestamp <= threshold
            for e in self._events
        )

    def needs_refresh(self, interval_hours: int = 12) -> bool:
        if self._last_fetch is None:
            return True
        return (datetime.utcnow() - self._last_fetch).total_seconds() > interval_hours * 3600
