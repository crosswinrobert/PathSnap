"""
Decoder Comparison: Classifier-only (Approach A) vs Classifier + HMM Viterbi (Approach B).
Member 3 Implementation - PathSnap Project, PES University.

Metrics Evaluated:
1. Observation-level Top-1 Accuracy
2. Macro F1-score
3. Route Continuity (% of consecutive transitions that are connected in graph)
4. Sequence-level Performance
5. Candidate Miss Rate
6. Inference Latency (ms / observation)
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score
from src.decoder.hmm import RoadNetwork, MapMatchingHMM
from src.decoder.viterbi import ViterbiDecoder


def evaluate_route_continuity(road_network: RoadNetwork, predicted_arcs: List[int]) -> float:
    """
    Compute percentage of consecutive transitions that are strictly connected in the road network.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    valid_transitions = 0
    total_transitions = len(predicted_arcs) - 1

    for t in range(total_transitions):
        prev_arc = predicted_arcs[t]
        curr_arc = predicted_arcs[t + 1]

        if prev_arc == curr_arc:
            valid_transitions += 1
        else:
            u_prev, v_prev = road_network.get_arc_nodes(prev_arc)
            u_curr, v_curr = road_network.get_arc_nodes(curr_arc)
            if v_prev == u_curr:
                valid_transitions += 1

    return float(valid_transitions / total_transitions)


def run_decoder_comparison(
    trips_data: List[Dict[str, Any]],
    output_csv_path: str = "experiments/decoder_comparison.csv",
    plot_path: str = "visualizations/routes/decoder_comparison.png"
) -> pd.DataFrame:
    """
    Runs comparison across evaluation trips.
    Each item in trips_data contains:
        trip_id: str
        road_network: RoadNetwork
        track: np.ndarray (T, 3)
        ground_truth: List[int] of true arc_id for each GPS point
        candidates: List[List[int]] of candidate arcs per step
        probabilities: List[np.ndarray] of classifier probabilities per candidate
    """
    results = []

    for trip_entry in trips_data:
        trip_id = trip_entry["trip_id"]
        rn = trip_entry["road_network"]
        track = trip_entry["track"]
        y_true = trip_entry["ground_truth"]
        candidates = trip_entry["candidates"]
        probs = trip_entry["probabilities"]
        T = len(track)

        # 1. Approach A: Classifier-only Top-1
        start_a = time.time()
        preds_a = []
        for cands, p in zip(candidates, probs):
            if len(cands) == 0:
                preds_a.append(0)
            else:
                best_idx = int(np.argmax(p))
                preds_a.append(cands[best_idx])
        latency_a = (time.time() - start_a) / max(T, 1) * 1000.0  # ms/obs

        # 2. Approach B: Classifier + HMM / Viterbi
        hmm = MapMatchingHMM(rn)
        decoder = ViterbiDecoder(hmm)
        start_b = time.time()
        preds_b = decoder.decode(track, candidates, probs)
        latency_b = (time.time() - start_b) / max(T, 1) * 1000.0  # ms/obs

        # Metrics
        acc_a = accuracy_score(y_true, preds_a)
        acc_b = accuracy_score(y_true, preds_b)

        f1_a = f1_score(y_true, preds_a, average="macro", zero_division=0)
        f1_b = f1_score(y_true, preds_b, average="macro", zero_division=0)

        cont_a = evaluate_route_continuity(rn, preds_a)
        cont_b = evaluate_route_continuity(rn, preds_b)

        # Candidate miss rate
        misses = sum(1 for yt, cands in zip(y_true, candidates) if yt not in cands)
        miss_rate = misses / max(T, 1)

        results.append({
            "trip_id": trip_id,
            "obs_count": T,
            "miss_rate": miss_rate,
            "clf_only_accuracy": acc_a,
            "hmm_viterbi_accuracy": acc_b,
            "clf_only_macro_f1": f1_a,
            "hmm_viterbi_macro_f1": f1_b,
            "clf_only_continuity": cont_a,
            "hmm_viterbi_continuity": cont_b,
            "clf_only_latency_ms": latency_a,
            "hmm_viterbi_latency_ms": latency_b
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved decoder comparison to {output_csv_path}")

    # Generate visual comparison plot
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Metric 1: Accuracy comparison
    mean_acc_a = df["clf_only_accuracy"].mean() * 100.0
    mean_acc_b = df["hmm_viterbi_accuracy"].mean() * 100.0
    axes[0].bar(["Classifier Only", "Classifier + HMM"], [mean_acc_a, mean_acc_b], color=["#e74c3c", "#2ecc71"])
    axes[0].set_ylabel("Top-1 Accuracy (%)")
    axes[0].set_title("Observation-Level Top-1 Accuracy")
    axes[0].set_ylim(0, 105)
    for i, v in enumerate([mean_acc_a, mean_acc_b]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    # Metric 2: Continuity comparison
    mean_cont_a = df["clf_only_continuity"].mean() * 100.0
    mean_cont_b = df["hmm_viterbi_continuity"].mean() * 100.0
    axes[1].bar(["Classifier Only", "Classifier + HMM"], [mean_cont_a, mean_cont_b], color=["#e74c3c", "#2ecc71"])
    axes[1].set_ylabel("Route Continuity (%)")
    axes[1].set_title("Topological Route Continuity")
    axes[1].set_ylim(0, 105)
    for i, v in enumerate([mean_cont_a, mean_cont_b]):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison visualization to {plot_path}")

    return df
