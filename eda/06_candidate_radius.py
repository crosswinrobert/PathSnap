"""Choose the candidate search radius R and cap K from TRAIN data only.

For each train trip we align the reference route to the GPS points (src/labels.py), generate
candidates at the largest radius, and measure how often the reference arc would be missed for each
(R, K). Only points whose aligned arc lies within RELIABLE_M of the GPS point are used, so trips with
incomplete reference routes do not distort the curve.

    python eda/06_candidate_radius.py  ->  experiments/eda/candidate_radius_tradeoff.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.candidates import CandidateGenerator
from src.config import DEFAULT_CFG as C
from src.data_io import DATA_DIR, list_trips, load_trip
from src.geo import LocalProjection
from src.labels import align_labels
from src.network import RoadNetwork
from src.preprocess import clean_track

OUT = "experiments/eda"
RADII = [10, 15, 20, 30, 40, 50, 75, 100]
KS = [3, 5, 10, 15, 20, 1000]
RELIABLE_M = 100.0


def one_trip(tid):
    tr = load_trip("train", tid, DATA_DIR)
    trk, _ = clean_track(tr.track)
    proj = LocalProjection.from_track(trk.lon.to_numpy(), trk.lat.to_numpy())
    xy = np.column_stack(proj.to_xy(trk.lon.to_numpy(), trk.lat.to_numpy()))
    net = RoadNetwork(tr.nodes, tr.arcs, proj)
    _, larc, ldist = align_labels(net, tr.route, xy, C["label_jump_penalty_m"], C["label_dist_clip_m"])
    cand = CandidateGenerator(net, xy, C["sample_spacing_m"], C["bbox_margin_m"]).generate(xy, max(RADII))
    ok = ldist <= RELIABLE_M
    hit = cand[cand["arc"].to_numpy() == larc[cand["pt"].to_numpy()]]
    rank = np.full(len(xy), 10**9)
    rank[hit["pt"].to_numpy()] = hit["rank"].to_numpy()
    rdist = np.full(len(xy), np.inf)
    rdist[hit["pt"].to_numpy()] = hit["dist"].to_numpy()
    rows = []
    for R in RADII:
        cnt = np.bincount(cand.loc[cand["dist"] <= R, "pt"], minlength=len(xy))
        for K in KS:
            miss = ~((rdist <= R) & (rank <= K))
            rows.append({"R": R, "K": K, "n_points": int(ok.sum()), "n_missed": int(miss[ok].sum()),
                         "sum_candidates": int(np.minimum(cnt, K)[ok].sum())})
    return rows


def main():
    trips = list_trips("train", DATA_DIR)
    with Pool(max(1, (__import__("os").cpu_count() or 2) - 1)) as pool:
        rows = [r for res in pool.map(one_trip, trips, chunksize=1) for r in res]
    df = pd.DataFrame(rows).groupby(["R", "K"], as_index=False).sum()
    df["miss_rate"] = df.n_missed / df.n_points
    df["mean_candidates"] = df.sum_candidates / df.n_points
    df = df[["R", "K", "n_points", "miss_rate", "mean_candidates"]]
    df.to_csv(f"{OUT}/candidate_radius_tradeoff.csv", index=False)
    print(df[df.K.isin([5, 10, 15, 1000])].pivot(index="R", columns="K", values="miss_rate").round(4).to_string())
    print(df[df.K == 1000].set_index("R")["mean_candidates"].round(1).to_string())


if __name__ == "__main__":
    main()
