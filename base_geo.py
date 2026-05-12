"""Approximate lat/lon for KC-135 operating bases.

Keyed by a KEYWORD that appears (case-insensitively, as a substring) in the raw
`Base` string in the maintenance data — the data uses inconsistent names like
`GREATER PITTSBURG`, `SCOTT AFB IL (ANG)`, `MCGUIRE ANG NJ`, so substring
matching on a stable keyword is more robust than exact-name lookup.

Coordinates are airfield-center, good enough for a fleet map. Order matters:
more-specific keywords should come before any that could be a substring of them.
"""
from __future__ import annotations

# keyword -> (display name, latitude, longitude)
_BASES: dict[str, tuple[str, float, float]] = {
    "FAIRCHILD": ("Fairchild AFB, WA", 47.615, -117.656),
    "MACDILL": ("MacDill AFB, FL", 27.849, -82.521),
    "MCCONNELL": ("McConnell AFB, KS", 37.622, -97.267),
    "ALTUS": ("Altus AFB, OK", 34.667, -99.267),
    "KADENA": ("Kadena AB, Okinawa, JP", 26.356, 127.769),
    "MILDENHALL": ("RAF Mildenhall, UK", 52.362, 0.486),
    "PITTSBURG": ("Pittsburgh ANGB, PA", 40.491, -80.233),
    "GRISSOM": ("Grissom ARB, IN", 40.648, -86.152),
    "MARCH": ("March ARB, CA", 33.881, -117.259),
    "SCOTT": ("Scott AFB, IL", 38.545, -89.835),
    "FORBES": ("Forbes Field ANGB (Topeka), KS", 38.951, -95.664),
    "BANGOR": ("Bangor ANGB, ME", 44.807, -68.828),
    "MITCHELL": ("Gen. Mitchell ANGB (Milwaukee), WI", 42.947, -87.897),
    "NIAGARA": ("Niagara Falls ARS, NY", 43.107, -78.946),
    "BIRMINGHAM": ("Birmingham ANGB, AL", 33.564, -86.754),
    "RICKENBACKER": ("Rickenbacker ANGB (Columbus), OH", 39.814, -82.928),
    "MCGHEE": ("McGhee Tyson ANGB (Knoxville), TN", 35.811, -83.994),
    "SKY HARBOR": ("Phoenix Sky Harbor ANGB, AZ", 33.434, -112.012),
    "LINCOLN": ("Lincoln ANGB, NE", 40.851, -96.759),
    "EIELSON": ("Eielson AFB, AK", 64.666, -147.099),
    "SIOUX": ("Sioux Gateway ANGB, IA", 42.403, -96.384),
    "SALT LAKE": ("Salt Lake City ANGB, UT", 40.786, -111.978),
    "SELFRIDGE": ("Selfridge ANGB, MI", 42.613, -82.836),
    "KEY FIELD": ("Key Field ANGB (Meridian), MS", 32.336, -88.751),
    "ANDREWS": ("JB Andrews, MD", 38.811, -76.867),
    "TINKER": ("Tinker AFB, OK", 35.415, -97.387),
    "MCGUIRE": ("JB McGuire-Dix-Lakehurst, NJ", 40.016, -74.593),
    "HICKAM": ("JB Pearl Harbor-Hickam, HI", 21.320, -157.953),
    "BEALE": ("Beale AFB, CA", 39.136, -121.437),
    "SEYMOUR": ("Seymour Johnson AFB, NC", 35.339, -77.961),
    "EDWARDS": ("Edwards AFB, CA", 34.905, -117.884),
    "PEASE": ("Pease ANGB, NH", 43.078, -70.823),
    "TRAVIS": ("Travis AFB, CA", 38.263, -121.927),
    "LITTLE ROCK": ("Little Rock AFB, AR", 34.917, -92.150),
    "DOVER": ("Dover AFB, DE", 39.130, -75.466),
    "YOKOTA": ("Yokota AB, JP", 35.748, 139.348),
    "RAMSTEIN": ("Ramstein AB, DE", 49.437, 7.600),
    "SPANGDAHLEM": ("Spangdahlem AB, DE", 49.973, 6.692),
    "INCIRLIK": ("Incirlik AB, TR", 37.002, 35.426),
    "CHARLESTON": ("JB Charleston, SC", 32.899, -80.041),
    "MCCHORD": ("JB Lewis-McChord, WA", 47.137, -122.476),
    "DYESS": ("Dyess AFB, TX", 32.421, -99.855),
    "ANCHORAGE": ("JB Elmendorf-Richardson, AK", 61.251, -149.807),
    "CHANNEL ISLAND": ("Channel Islands ANGS, CA", 34.120, -119.122),
    "JACKSON": ("Jackson ANGB, MS", 32.321, -90.077),
    "STEWART": ("Stewart ANGB, NY", 41.504, -74.105),
    "QUONSET": ("Quonset State ANGB, RI", 41.597, -71.412),
    "MINN": ("Minneapolis-St. Paul ARS, MN", 44.880, -93.207),
    "SAVANNAH": ("Savannah ANGB, GA", 32.128, -81.202),
    "CARSWELL": ("NAS Fort Worth JRB (Carswell), TX", 32.769, -97.441),
    "KEESLER": ("Keesler AFB, MS", 30.412, -88.924),
    "MARTINSBURG": ("Martinsburg ANGB, WV", 39.402, -77.985),
    "WRIGHT": ("Wright-Patterson AFB, OH", 39.826, -84.048),
    "DOBBINS": ("Dobbins ARB, GA", 33.915, -84.516),
    "POPE": ("Pope Field, NC", 35.171, -79.014),
    "RENO": ("Reno ANGB, NV", 39.499, -119.768),
    "ROSECRANS": ("Rosecrans ANGB (St. Joseph), MO", 39.772, -94.910),
}


def geolocate(base_raw) -> tuple[str, float, float] | None:
    """Map a raw `Base` string to (display_name, lat, lon), or None if unknown."""
    if not isinstance(base_raw, str) or not base_raw.strip():
        return None
    s = base_raw.upper()
    for kw, entry in _BASES.items():
        if kw in s:
            return entry
    return None
