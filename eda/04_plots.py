"""EDA figures (PNG, 150 dpi). Run after 01_inventory.py, build_dataset.py and 06_candidate_radius.py.

    python eda/04_plots.py  ->  visualizations/eda/*.png
Each figure title names the partition(s) it was computed on.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_io import DATA_DIR, load_table, load_trip

EDA, FEAT, FIG = "experiments/eda", "data/processed", "visualizations/eda"
COL = {"train": "#1f77b4", "student_test": "#ff7f0e"}
os.makedirs(FIG, exist_ok=True)
inv = pd.read_csv(f"{EDA}/dataset_inventory.csv", dtype={"trip_id": str})


def save(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=150); plt.close()


# 1 dataset overview
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
inv.groupby("partition").size().reindex(["train", "student_test"]).plot.bar(ax=ax[0], color=[COL["train"], COL["student_test"]])
ax[0].set_title("Trips per partition"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
for p, d in inv.groupby("partition"):
    ax[1].hist(np.log10(d.n_gps), bins=20, alpha=0.6, label=p, color=COL[p])
    ax[2].hist(d.track_length_km, bins=20, alpha=0.6, label=p, color=COL[p])
ax[1].set_title("GPS observations per trip"); ax[1].set_xlabel("log10(number of GPS points)"); ax[1].legend()
ax[2].set_title("Travelled distance per trip"); ax[2].set_xlabel("km"); ax[2].legend()
save("01_dataset_overview")

# 2 timestamp analysis (all consecutive steps)
dts = {p: [] for p in COL}
for _, r in inv.iterrows():
    t = load_trip(r.partition, r.trip_id, DATA_DIR, load_route=False).track["t"].to_numpy()
    dts[r.partition].append(np.diff(t))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p in COL:
    d = np.concatenate(dts[p])
    ax[0].hist(np.clip(d, 0, 20), bins=np.arange(0, 21, 0.5), alpha=0.6, label=f"{p} ({np.mean(np.abs(d-1)<0.05)*100:.1f}% = 1 s)", color=COL[p], density=True)
    ax[1].hist(np.log10(d[d > 1.05]), bins=30, alpha=0.6, label=p, color=COL[p])
ax[0].set_yscale("log"); ax[0].set_title("Sampling interval (clipped at 20 s)"); ax[0].set_xlabel("seconds"); ax[0].legend()
ax[1].set_title("Gaps > 1 s"); ax[1].set_xlabel("log10(gap seconds)"); ax[1].set_ylabel("count"); ax[1].legend()
s = inv.sort_values("frac_dt_1s").reset_index(drop=True)
ax[2].bar(range(len(s)), s.frac_dt_1s, color=[COL[p] for p in s.partition])
ax[2].set_title("Fraction of 1 s steps per trip (sorted)"); ax[2].set_xlabel("trip rank"); ax[2].set_ylim(0, 1.02)
save("02_timestamp_analysis")

# 3 geographic spread
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for p, d in inv.groupby("partition"):
    ax[0].scatter(d.lon_median, d.lat_median, s=22, alpha=0.8, label=p, color=COL[p])
    ax[1].scatter(d.bbox_diag_km, d.n_gps, s=22, alpha=0.8, label=p, color=COL[p])
ax[0].set_title("Trip locations (median GPS position)"); ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude"); ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_title("Geographic extent vs. #GPS points")
ax[1].set_xlabel("bounding-box diagonal (km)"); ax[1].set_ylabel("GPS points")
save("03_geographic_extent")

# 4 road networks
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p, d in inv.groupby("partition"):
    ax[0].scatter(d.n_nodes, d.n_arcs, s=22, alpha=0.8, label=p, color=COL[p])
    ax[1].scatter(d.n_weak_components, d.largest_weak_comp_share, s=22, alpha=0.8, label=p, color=COL[p])
    ax[2].scatter(d.n_gps, d.n_route, s=22, alpha=0.8, label=p, color=COL[p])
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_title("Road-network size"); ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].legend()
ax[1].set_xscale("log"); ax[1].set_title("Connectivity: weak components"); ax[1].set_xlabel("# weakly connected components"); ax[1].set_ylabel("share of nodes in largest")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_title("GPS points vs. reference-route arcs"); ax[2].set_xlabel("GPS points"); ax[2].set_ylabel("route arcs")
save("04_road_network")

# 5 labels + candidates (needs build_dataset.py outputs)
try:
    pts = {p: load_table(f"{FEAT}/points_{p}") for p in COL}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, d in pts.items():
        ax[0].hist(np.clip(d.label_dist.dropna(), 0, 200), bins=60, alpha=0.6, label=f"{p} ({d.label_valid.mean()*100:.1f}% valid)", color=COL[p])
        ax[1].hist(d.n_cand, bins=np.arange(0, 17), alpha=0.6, label=p, color=COL[p], density=True)
    ax[0].set_yscale("log"); ax[0].axvline(50, color="k", ls="--", lw=1); ax[0].set_title("Distance GPS point -> aligned reference arc")
    ax[0].set_xlabel("metres (clipped at 200)"); ax[0].legend()
    ax[1].set_title("Candidates per GPS observation"); ax[1].set_xlabel("# candidates"); ax[1].legend()
    bs = pd.concat([pd.read_csv(f"{FEAT}/build_summary_{p}.csv", dtype={"trip_id": str}) for p in COL])
    for p, d in bs.groupby("partition"):
        ax[2].scatter(d.n_points, d.nearest_arc_accuracy_valid, s=22, alpha=0.8, label=p, color=COL[p])
    ax[2].set_xscale("log"); ax[2].set_ylim(0, 1); ax[2].set_title("Baseline: 'nearest arc' accuracy per trip (valid labels)")
    ax[2].set_xlabel("GPS points"); ax[2].legend()
    save("05_labels_and_candidates")
except FileNotFoundError:
    print("skip fig 05 (run build_dataset.py first)")

# 6 radius / K trade-off (train only)
try:
    t = pd.read_csv(f"{EDA}/candidate_radius_tradeoff.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for K, d in t.groupby("K"):
        ax[0].plot(d.R, d.miss_rate * 100, marker="o", label="K = all" if K == 1000 else f"K = {K}")
    ax[0].axvline(50, color="k", ls="--", lw=1); ax[0].set_xlabel("search radius R (m)"); ax[0].set_ylabel("candidate miss rate (%)")
    ax[0].set_title("Reference arc missing from candidate set (train)"); ax[0].legend()
    d = t[t.K == 1000]; ax[1].plot(d.R, d.mean_candidates, marker="o", color="C3")
    ax[1].axvline(50, color="k", ls="--", lw=1); ax[1].set_xlabel("search radius R (m)"); ax[1].set_ylabel("mean candidates per GPS point")
    ax[1].set_title("Candidate-set size vs. radius (train)")
    save("06_candidate_radius_tradeoff")
except FileNotFoundError:
    print("skip fig 06 (run 06_candidate_radius.py first)")
print("figures in", FIG)
