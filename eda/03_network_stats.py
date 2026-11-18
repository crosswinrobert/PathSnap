"""Road-network size and connectivity tables.

    python eda/03_network_stats.py  ->  experiments/eda/network_connectivity_summary.csv
                                        experiments/eda/network_degree_distribution.csv
Reads dataset_inventory.csv (run 01 first) and recomputes node degree distributions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.data_io import DATA_DIR, load_trip

OUT = "experiments/eda"


def main():
    inv = pd.read_csv(f"{OUT}/dataset_inventory.csv", dtype={"trip_id": str})
    cols = ["n_nodes", "n_arcs", "n_isolated_nodes", "n_self_loops", "n_duplicate_arcs",
            "n_weak_components", "largest_weak_comp_share", "n_strong_components",
            "largest_strong_comp_share", "route_n_weak_components"]
    summ = inv.groupby("partition")[cols].agg(["min", "median", "mean", "max"]).T
    summ.to_csv(f"{OUT}/network_connectivity_summary.csv")
    print(summ.round(3).to_string())

    rows = []
    for _, r in inv.iterrows():
        tr = load_trip(r.partition, r.trip_id, DATA_DIR, load_route=False)
        outd = np.bincount(tr.arcs.u, minlength=len(tr.nodes))
        for k, c in zip(*np.unique(outd, return_counts=True)):
            rows.append({"partition": r.partition, "trip_id": r.trip_id, "out_degree": int(k), "n_nodes": int(c)})
    pd.DataFrame(rows).to_csv(f"{OUT}/network_degree_distribution.csv", index=False)
    print("\ntrips whose road network has more than one weakly connected component:",
          int((inv.n_weak_components > 1).sum()), "of", len(inv), "trips")
    print("trips whose reference route spans more than one weak component:",
          int((inv.route_n_weak_components > 1).sum()))


if __name__ == "__main__":
    main()
