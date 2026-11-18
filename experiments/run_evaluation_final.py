"""Classifier-only vs classifier + HMM/Viterbi on the 10 student-test trips."""

import sys
import numpy as np
import joblib

from src.data_io import load_trip, list_trips
from src.config import load_cfg
from src.features import build_trip_tables, FEATURE_COLS
from src.decoder.hmm import RoadNetwork
from src.evaluation.compare_decoders import run_decoder_comparison
from experiments.member3_experiments import project_track_to_route

MODEL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "artifacts/models/member2_random_forest.joblib"
)
CFG_PATH = "artifacts/configs/feature_config.json"


def main():
    model = joblib.load(MODEL_PATH)

    assert model.n_features_in_ == len(FEATURE_COLS), (
        model.n_features_in_,
        len(FEATURE_COLS),
    )

    pos = list(model.classes_).index(1)
    cfg = load_cfg(CFG_PATH)
    trips_data = []

    for trip_id in list_trips("student_test"):
        trip = load_trip("student_test", trip_id, load_route=False)

        cand_df, point_df, _ = build_trip_tables(
            trip, cfg, with_labels=False
        )

        scores = model.predict_proba(
            cand_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        )[:, pos]

        scored = cand_df[["obs_idx", "arc_id"]].copy()
        scored["score"] = scores

        grouped = {
            int(k): g
            for k, g in scored.groupby("obs_idx", sort=False)
        }

        pts = point_df.sort_values("obs_idx")
        candidates, probabilities = [], []

        for obs_idx in pts["obs_idx"]:
            group = grouped.get(int(obs_idx))

            candidates.append(
                [] if group is None
                else group["arc_id"].astype(int).tolist()
            )

            probabilities.append(
                np.array([]) if group is None
                else group["score"].to_numpy(dtype=float)
            )

        ref = load_trip("student_test", trip_id, load_route=True)

        rn = RoadNetwork(
            ref.nodes[["lon", "lat"]].to_numpy(dtype=float),
            ref.arcs[["u", "v"]].to_numpy(dtype=np.int64),
        )

        track = ref.track[["lon", "lat", "t"]].to_numpy(dtype=float)

        ground_truth = project_track_to_route(
            track, rn, ref.route.tolist()
        )

        trips_data.append({
            "trip_id": trip_id,
            "road_network": rn,
            "track": pts[["lon", "lat", "t"]].to_numpy(dtype=float),
            "ground_truth": ground_truth,
            "candidates": candidates,
            "probabilities": probabilities,
        })

        print("done", trip_id, len(ground_truth), flush=True)

    df = run_decoder_comparison(
        trips_data=trips_data,
        output_csv_path="experiments/decoder_comparison_final.csv",
        plot_path="visualizations/routes/decoder_comparison_final.png",
    )

    cols = [
        "clf_only_accuracy",
        "hmm_viterbi_accuracy",
        "clf_only_macro_f1",
        "hmm_viterbi_macro_f1",
        "clf_only_continuity",
        "hmm_viterbi_continuity",
        "miss_rate",
    ]

    print(df[["trip_id"] + cols].round(4).to_string(index=False))
    print("\nMean metrics:")
    print(df[cols].mean().round(4))


if __name__ == "__main__":
    main()
