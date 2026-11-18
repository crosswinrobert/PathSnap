"""Raw-file loading for PathSnap trips.

File formats (verified on the real data, no headers, whitespace separated):
  .track  : lon  lat  t(seconds)     (tab separated, CRLF line endings)
  .nodes  : lon  lat                 (row index = node_id, 0-based)
  .arcs   : from_node  to_node       (row index = arc_id, 0-based, directed)
  .route  : arc_id                   (chronological reference route; training/eval only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("PATHSNAP_DATA", "data/raw")
PARTITIONS = ("train", "student_test")


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame            # lon, lat, t
    nodes: pd.DataFrame            # lon, lat
    arcs: pd.DataFrame             # u, v
    route: Optional[np.ndarray]    # arc ids, or None when unavailable


def list_trips(partition: str, data_dir: str = DATA_DIR):
    d = os.path.join(data_dir, partition)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _read(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_trip(partition: str, trip_id: str, data_dir: str = DATA_DIR,
              load_route: bool = True) -> Trip:
    base = os.path.join(data_dir, partition, trip_id, trip_id)
    track = _read(base + ".track", ["lon", "lat", "t"])
    nodes = _read(base + ".nodes", ["lon", "lat"])
    arcs = _read(base + ".arcs", ["u", "v"]).astype(np.int64)
    route = None
    rp = base + ".route"
    if load_route and os.path.exists(rp):
        route = pd.read_csv(rp, header=None)[0].to_numpy(dtype=np.int64)
    return Trip(trip_id, partition, track, nodes, arcs, route)


def file_info(partition: str, trip_id: str, data_dir: str = DATA_DIR):
    """Availability and size (bytes) of each raw file of a trip."""
    base = os.path.join(data_dir, partition, trip_id, trip_id)
    out = {}
    for ext in ("track", "nodes", "arcs", "route"):
        p = f"{base}.{ext}"
        out[f"has_{ext}"] = os.path.exists(p)
        out[f"size_{ext}"] = os.path.getsize(p) if os.path.exists(p) else 0
    return out


def save_table(df: pd.DataFrame, path_base: str) -> str:
    """Parquet if pyarrow is installed, otherwise gzip CSV. Returns the written path."""
    os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        p = path_base + ".parquet"
        df.to_parquet(p, index=False)
    except ImportError:
        p = path_base + ".csv.gz"
        df.to_csv(p, index=False)
    return p


def load_table(path_base: str) -> pd.DataFrame:
    for ext, fn in ((".parquet", pd.read_parquet), (".csv.gz", pd.read_csv)):
        if os.path.exists(path_base + ext):
            return fn(path_base + ext)
    raise FileNotFoundError(path_base)
