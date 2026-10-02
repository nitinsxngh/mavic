from __future__ import annotations

import httpx

from typing import Optional

from app.context.models import LocationContext


def reverse_geocode(lat: float, lng: float, accuracy_m: Optional[float] = None) -> LocationContext:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lng, "format": "jsonv2"}
    headers = {"User-Agent": "MavicAgentBuddy/1.0 (local-dev)"}
    city = state = country = formatted = None
    try:
        with httpx.Client(timeout=6.0) as client:
            data = client.get(url, params=params, headers=headers).json()
        address = data.get("address") or {}
        city = address.get("city") or address.get("town") or address.get("village") or address.get("suburb")
        state = address.get("state")
        country = address.get("country")
        formatted = data.get("display_name")
    except Exception:
        formatted = f"{lat:.5f}, {lng:.5f}"

    return LocationContext(
        latitude=lat,
        longitude=lng,
        accuracy_m=accuracy_m,
        city=city,
        state=state,
        country=country,
        formatted=formatted,
        source="gps",
    )
