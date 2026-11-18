"""Build the model-ready tables for a partition.

    python build_dataset.py --partition train --data-dir data --out data/processed
    python build_dataset.py --partition student_test

Outputs (parquet if pyarrow is installed, else .csv.gz):
    features_<partition>   one row per (GPS observation, candidate arc), with label columns
    points_<partition>     one row per GPS observation (label, label distance, validity, #candidates)
    build_summary_<partition>.csv   per-trip diagnostics (candidate miss rate, nearest-arc baseline, ...)
    feature_config.json    parameters used (radius, max candidates, ...)

The same `build_trip_tables` is what the API calls at inference time with with_labels=False.
"""
import argparse
import os
import time
from multiprocessing import Pool

import pandas as pd

from src.config import DEFAULT_CFG, save_cfg
from src.data_io import list_trips, load_trip, save_table
from src.features import build_trip_tables


def _work(args):
    partition, tid, data_dir, cfg = args
    cand, pts, meta = build_trip_tables(load_trip(partition, tid, data_dir), cfg, with_labels=True)
    return cand, pts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True, choices=["train", "student_test"])
    ap.add_argument("--data-dir", default=os.environ.get("PATHSNAP_DATA", "data/raw"))
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    trips = list_trips(a.partition, a.data_dir)
    t0 = time.time()
    jobs = [(a.partition, t, a.data_dir, DEFAULT_CFG) for t in trips]
    with Pool(a.workers) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    cand = pd.concat([r[0] for r in res], ignore_index=True)
    pts = pd.concat([r[1] for r in res], ignore_index=True)
    summ = pd.DataFrame([r[2] for r in res])

    os.makedirs(a.out, exist_ok=True)
    print("wrote", save_table(cand, os.path.join(a.out, f"features_{a.partition}")))
    print("wrote", save_table(pts, os.path.join(a.out, f"points_{a.partition}")))
    summ.to_csv(os.path.join(a.out, f"build_summary_{a.partition}.csv"), index=False)
    save_cfg(DEFAULT_CFG, "artifacts/configs/feature_config.json")
    print(f"{a.partition}: {len(trips)} trips, {len(pts)} GPS points, {len(cand)} candidate rows, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
