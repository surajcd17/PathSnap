"""Geographic helpers: per-trip metric projection and point/segment geometry.

Trips are spread over different regions (e.g. ~8.6E/49.8N and ~11.0E/47.7N), so each
trip is projected into the UTM zone of its own GPS centroid. Inside one zone UTM
distortion is < 0.1 %, which is negligible compared with GPS noise (metres).
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def to_xy(self, lon, lat) -> np.ndarray:
        x, y = self._fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.column_stack([x, y])

    def to_lonlat(self, x, y) -> np.ndarray:
        lon, lat = self._inv.transform(np.asarray(x), np.asarray(y))
        return np.column_stack([lon, lat])


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p to segments a->b (all (n,2), broadcast row-wise).

    Returns (distance, t) where t in [0,1] is the projection fraction along a->b.
    """
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    t = np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return np.hypot(*(p - proj).T), t


def bearing(dx, dy) -> np.ndarray:
    """Compass-style bearing in degrees [0, 360) of vector (dx, dy) in a metric plane."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b) -> np.ndarray:
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)
