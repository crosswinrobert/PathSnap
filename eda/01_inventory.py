"""Dataset inventory + data-quality table: one row per trip, both partitions.

    python eda/01_inventory.py            ->  experiments/eda/dataset_inventory.csv
                                              experiments/eda/inventory_summary_by_partition.csv
Covers: trip counts, file availability/sizes, GPS counts, timestamp ordering and gaps, missing values,
geographic extent, road-network size and connectivity (components, isolated nodes), reference-route
structure and continuity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data_io import DATA_DIR, PARTITIONS, file_info, list_trips, load_trip
from src.geo import haversine_m

OUT = "experiments/eda"


def trip_row(partition, tid):
    tr = load_trip(partition, tid, DATA_DIR)
    trk, nodes, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
    lon, lat, t = trk["lon"].to_numpy(), trk["lat"].to_numpy(), trk["t"].to_numpy()
    dt = np.diff(t)
    n_nodes, n_arcs = len(nodes), len(arcs)
    u, v = arcs["u"].to_numpy(), arcs["v"].to_numpy()

    g = coo_matrix((np.ones(n_arcs), (u, v)), shape=(n_nodes, n_nodes))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    w_sizes, s_sizes = np.bincount(lab_w), np.bincount(lab_s)

    r = route
    r_conn = float(np.mean(v[r[:-1]] == u[r[1:]])) if len(r) > 1 else np.nan
    pts_in_bbox = float(np.mean((lon >= nodes.lon.min()) & (lon <= nodes.lon.max())
                                & (lat >= nodes.lat.min()) & (lat <= nodes.lat.max())))
    return {
        "partition": partition, "trip_id": tid, **file_info(partition, tid, DATA_DIR),
        # GPS observations
        "n_gps": len(trk), "n_missing_values": int(trk.isna().sum().sum()),
        "n_invalid_coords": int(((trk.lon.abs() > 180) | (trk.lat.abs() > 90)).sum()),
        "n_repeated_points": int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum()),
        "duration_s": float(t[-1] - t[0]), "timestamps_monotonic": bool((dt > 0).all()),
        "n_duplicate_timestamps": int((dt == 0).sum()),
        "frac_dt_1s": float(np.mean(np.abs(dt - 1) < 0.05)), "dt_median_s": float(np.median(dt)),
        "dt_max_s": float(dt.max()), "n_gaps_gt5s": int((dt > 5).sum()),
        "n_gaps_gt30s": int((dt > 30).sum()), "n_gaps_gt300s": int((dt > 300).sum()),
        # geography
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "lon_median": float(np.median(lon)), "lat_median": float(np.median(lat)),
        "bbox_diag_km": float(haversine_m(lon.min(), lat.min(), lon.max(), lat.max()) / 1000),
        "track_length_km": float(haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:]).sum() / 1000),
        "gps_in_network_bbox": pts_in_bbox,
        # road network
        "n_nodes": n_nodes, "n_arcs": n_arcs, "n_self_loops": int((u == v).sum()),
        "n_duplicate_arcs": int(arcs.duplicated().sum()),
        "n_isolated_nodes": int(n_nodes - len(np.unique(np.r_[u, v]))),
        "n_weak_components": int(n_weak), "largest_weak_comp_share": float(w_sizes.max() / n_nodes),
        "n_strong_components": int(n_strong), "largest_strong_comp_share": float(s_sizes.max() / n_nodes),
        # reference route
        "n_route": len(r), "n_route_unique": int(len(np.unique(r))),
        "route_arc_ids_valid": bool(r.min() >= 0 and r.max() < n_arcs),
        "route_continuity": r_conn,                        # share of consecutive route arcs that connect
        "route_n_weak_components": int(len(np.unique(lab_w[u[r]]))),
        "gps_per_route_arc": float(len(trk) / len(r)),
    }


def main():
    rows = []
    for p in PARTITIONS:
        for tid in list_trips(p, DATA_DIR):
            rows.append(trip_row(p, tid))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/dataset_inventory.csv", index=False)
    num = ["n_gps", "duration_s", "frac_dt_1s", "dt_max_s", "bbox_diag_km", "track_length_km",
           "n_nodes", "n_arcs", "n_route", "route_continuity", "n_weak_components"]
    summ = df.groupby("partition")[num].describe().T
    summ.to_csv(f"{OUT}/inventory_summary_by_partition.csv")
    print(df.groupby("partition").size().rename("n_trips").to_string())
    print(summ.round(3).to_string())
    print(f"\nwrote {OUT}/dataset_inventory.csv ({len(df)} trips)")


if __name__ == "__main__":
    main()
