"""Raw-data loading for PathSnap trips.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node id)
  <id>.arcs   from_node to_node  (row index = arc id, directed)
  <id>.route  arc_id             (reference arc sequence; optional / server-side only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Student folder"


def data_root() -> Path:
    return Path(os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT))


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    base = (root or data_root()) / partition
    return sorted(p.name for p in base.iterdir() if p.is_dir())


@dataclass
class RawTrip:
    trip_id: str
    track: pd.DataFrame          # columns lon, lat, t
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from, to (int64)
    route: np.ndarray | None     # (R,) arc ids, or None if not available


def _read_table(path: Path, names, dtype=float) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def load_trip(trip_dir: Path | str, with_route: bool = True) -> RawTrip:
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    track = _read_table(trip_dir / f"{tid}.track", ["lon", "lat", "t"])
    nodes = _read_table(trip_dir / f"{tid}.nodes", ["lon", "lat"]).to_numpy()
    arcs = _read_table(trip_dir / f"{tid}.arcs", ["u", "v"], dtype=np.int64).to_numpy()
    route = None
    rpath = trip_dir / f"{tid}.route"
    if with_route and rpath.exists():
        route = _read_table(rpath, ["arc"], dtype=np.int64)["arc"].to_numpy()
    return RawTrip(tid, track, nodes, arcs, route)


def clean_track(track: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows, sort by time, remove duplicate timestamps.

    Returns the cleaned track (with a column `orig_idx` pointing to the raw row)
    and a dict describing what was removed.
    """
    df = track.copy()
    df["orig_idx"] = np.arange(len(df))
    n0 = len(df)
    bad = df[["lon", "lat", "t"]].isna().any(axis=1)
    bad |= ~df["lon"].between(-180, 180) | ~df["lat"].between(-90, 90)
    bad |= (df["lon"] == 0) & (df["lat"] == 0)
    df = df[~bad]
    n_invalid = int(bad.sum())
    non_monotone = int((np.diff(df["t"].to_numpy()) <= 0).sum())
    df = df.sort_values("t", kind="stable").drop_duplicates("t", keep="first")
    info = {
        "n_raw": n0,
        "n_invalid": n_invalid,
        "n_non_monotone_steps": non_monotone,
        "n_dropped_duplicate_t": n0 - n_invalid - len(df),
        "n_clean": len(df),
    }
    return df.reset_index(drop=True), info
