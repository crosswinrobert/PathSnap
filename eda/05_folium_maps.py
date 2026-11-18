"""Interactive Folium maps (raw GPS, road network, reference route).

    python eda/05_folium_maps.py                       # 3 representative trips per partition
    python eda/05_folium_maps.py --partition train --trips 00000002 00000005
Output: visualizations/eda/maps/<partition>_<trip>.html   (open in a browser; take a screenshot for the PDF)
Requires: pip install folium
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

import pandas as pd

from src.data_io import DATA_DIR, list_trips, load_trip
from src.viz import make_trip_map

OUT = "visualizations/eda/maps"


def pick_default(partition):
    inv = pd.read_csv("experiments/eda/dataset_inventory.csv", dtype={"trip_id": str})
    inv = inv[inv.partition == partition].sort_values("n_gps")
    return [inv.trip_id.iloc[0], inv.trip_id.iloc[len(inv) // 2], inv.trip_id.iloc[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["train", "student_test"])
    ap.add_argument("--trips", nargs="*")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    parts = [a.partition] if a.partition else ["train", "student_test"]
    for p in parts:
        for tid in (a.trips or pick_default(p)):
            trip = load_trip(p, tid, DATA_DIR)
            path = f"{OUT}/{p}_{tid}.html"
            make_trip_map(trip, path)
            print("wrote", path)


if __name__ == "__main__":
    main()
