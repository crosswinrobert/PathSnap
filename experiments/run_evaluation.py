# Fast Evaluator comparison script on all 10 student testing trips.
import os, glob, time, joblib
import numpy as np
import pandas as pd
from src.decoder.hmm import RoadNetwork, MapMatchingHMM
from src.decoder.viterbi import ViterbiDecoder
from src.features.candidate_features import extract_trip_candidates
from src.evaluation.compare_decoders import run_decoder_comparison
from experiments.member3_experiments import project_track_to_route

def main():
    base_dir = "/Users/amit/PathSnap"
    test_dir = os.path.join(base_dir, "data/raw/student_test")
    model_path = os.path.join(base_dir, "artifacts/models/best_classifier.joblib")
    print("Loading refitted best classifier from:", model_path, flush=True)
    best_classifier = joblib.load(model_path)
    best_classifier.set_params(classifier__n_jobs=1)
    test_trip_folders = sorted(glob.glob(os.path.join(test_dir, "*")))
    print("Found", len(test_trip_folders), "student testing trips.", flush=True)
    trips_eval_data = []
    for trip_path in test_trip_folders:
        trip_id = os.path.basename(trip_path)
        track_path = os.path.join(trip_path, trip_id + ".track")
        nodes_path = os.path.join(trip_path, trip_id + ".nodes")
        arcs_path = os.path.join(trip_path, trip_id + ".arcs")
        route_path = os.path.join(trip_path, trip_id + ".route")
        if not all(os.path.exists(p) for p in [track_path, nodes_path, arcs_path, route_path]):
            continue
        t0 = time.time()
        rn = RoadNetwork.from_files(nodes_path, arcs_path)
        track = np.loadtxt(track_path)
        route = [int(line.strip()) for line in open(route_path)]
        gt_arcs = project_track_to_route(track, rn, route)
        cands_per_step, feats_per_step = extract_trip_candidates(rn, track, radius_meters=60.0)
        all_lens = [len(f) for f in feats_per_step]
        valid_feats = [f for f in feats_per_step if len(f) > 0]
        if valid_feats:
            X_all = np.vstack(valid_feats)
            probs_all = best_classifier.predict_proba(X_all)[:, 1]
            probs_per_step = []
            offset = 0
            for l in all_lens:
                if l == 0:
                    probs_per_step.append(np.array([]))
                else:
                    probs_per_step.append(probs_all[offset:offset + l])
                    offset += l
        else:
            probs_per_step = [np.array([]) for _ in all_lens]
        trips_eval_data.append({
            "trip_id": trip_id,
            "road_network": rn,
            "track": track,
            "ground_truth": gt_arcs,
            "candidates": cands_per_step,
            "probabilities": probs_per_step
        })
        print("Trip", trip_id, "preprocessed in", round(time.time() - t0, 2), "s (Points:", len(track), ")", flush=True)
    print("")
    print("Running full decoder comparison (Approach A vs Approach B)...", flush=True)
    t_start = time.time()
    df_results = run_decoder_comparison(
        trips_data=trips_eval_data,
        output_csv_path=os.path.join(base_dir, "experiments/decoder_comparison.csv"),
        plot_path=os.path.join(base_dir, "visualizations/routes/decoder_comparison.png")
    )
    print("Comparison finished in", round(time.time() - t_start, 2), "s", flush=True)
    print("")
    print("=" * 80)
    print("FINAL COMPARISON SUMMARY TABLE: CLASSIFIER-ONLY vs CLASSIFIER + HMM/VITERBI")
    print("=" * 80)
    cols = [
        "trip_id", "obs_count", "miss_rate",
        "clf_only_accuracy", "hmm_viterbi_accuracy",
        "clf_only_continuity", "hmm_viterbi_continuity",
        "hmm_viterbi_latency_ms"
    ]
    print(df_results[cols].to_string(index=False), flush=True)
    print("")
    print("=" * 80)
    print("AGGREGATE EXPERIMENT FINDINGS")
    print("=" * 80)
    print("Classifier-Only Mean Top-1 Accuracy: ", round(df_results["clf_only_accuracy"].mean() * 100, 2), "%")
    print("Classifier + HMM Mean Top-1 Accuracy:", round(df_results["hmm_viterbi_accuracy"].mean() * 100, 2), "%")
    print("Classifier-Only Mean Continuity:    ", round(df_results["clf_only_continuity"].mean() * 100, 2), "%")
    print("Classifier + HMM Mean Continuity:   ", round(df_results["hmm_viterbi_continuity"].mean() * 100, 2), "%")
    print("Mean Candidate Miss Rate:           ", round(df_results["miss_rate"].mean() * 100, 2), "%")
    print("Mean HMM/Viterbi Inference Latency: ", round(df_results["hmm_viterbi_latency_ms"].mean(), 3), "ms / obs")
    print("=" * 80)

if __name__ == "__main__":
    main()
