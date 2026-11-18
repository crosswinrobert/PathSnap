"""Geometry helpers.

Projection choice (justified for the report)
--------------------------------------------
Longitude/latitude are angles, so Euclidean distance in degrees is wrong (a degree of longitude
shrinks with cos(latitude)). Every trip in this dataset has its OWN road network and the trips are
spread over the whole world, so no single global projection is suitable. For each trip we use a
local equirectangular (plate carree) projection centred on the median of its GPS track:
    x = R * cos(lat0) * (lon - lon0),   y = R * (lat - lat0)      (radians, R = 6 371 008.8 m)
Distortion grows with distance from the centre but is well below 1 % over the few-km scale on
which candidate arcs are searched. The per-trip extent is reported in the inventory
(eda/01_inventory.py) so this assumption can be checked.
"""
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self._k = np.cos(np.radians(self.lat0))

    @classmethod
    def from_track(cls, lon, lat):
        return cls(np.median(lon), np.median(lat))

    def to_xy(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        x = EARTH_R * self._k * np.radians(lon - self.lon0)
        y = EARTH_R * np.radians(lat - self.lat0)
        return x, y


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment_dist(px, py, ax, ay, bx, by):
    """Distance from points to segments a->b (all arrays broadcastable).
    Returns (distance, frac) with frac in [0,1] = position of the projection along the segment."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.where(l2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a displacement in projected metres."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)
