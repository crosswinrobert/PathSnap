# Master Reproducible Experiment Script for Member 3
import os
import glob
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from src.decoder.hmm import RoadNetwork, MapMatchingHMM
from src.decoder.viterbi import ViterbiDecoder
from src.features.candidate_features import extract_trip_candidates, point_to_segment_projection
from src.models.baseline_classifier import build_candidate_classifier
from src.optimization.bayesian_optimization import BayesianOptimizer
from src.evaluation.compare_decoders import run_decoder_comparison, evaluate_route_continuity


def project_track_to_route(track, road_network, route_arcs):
    nodes = road_network.nodes
    arcs = road_network.arcs
    matched_arcs = []
    r_idx = 0

    for pt in track:
        p_lon, p_lat = pt[0], pt[1]
        best_d = float('inf')
        best_arc = route_arcs[r_idx]
        best_r = r_idx

        start_r = max(0, r_idx - 1)
        end_r = min(len(route_arcs), r_idx + 8)

        for r in range(start_r, end_r):
            arc_id = route_arcs[r]
            u, v = road_network.get_arc_nodes(arc_id)
            d, _, _ = point_to_segment_projection(
                p_lon, p_lat,
                nodes[u, 0], nodes[u, 1],
                nodes[v, 0], nodes[v, 1]
            )
            if d < best_d:
                best_d = d
                best_arc = arc_id
                best_r = r

        matched_arcs.append(best_arc)
        r_idx = best_r

    return matched_arcs


def main():
    print('=' * 70)
    print('PATHSNAP HACKATHON - MEMBER 3 OPTIMISATION & ROUTE RECONSTRUCTION')
    print('=' * 70)

    base_dir = '/Users/amit/PathSnap'
    train_dir = os.path.join(base_dir, 'data/raw/train')
    test_dir = os.path.join(base_dir, 'data/raw/student_test')

    train_trip_folders = sorted(glob.glob(os.path.join(train_dir, '*')))
    print('Found', len(train_trip_folders), 'training trips in', train_dir)

    selected_train_trips = train_trip_folders[:10]
    print('Processing', len(selected_train_trips), 'trips for feature extraction & Optuna tuning...')

    all_X = []
    all_y = []
    all_groups = []

    for trip_path in selected_train_trips:
        trip_id = os.path.basename(trip_path)
        track_path = os.path.join(trip_path, trip_id + '.track')
        nodes_path = os.path.join(trip_path, trip_id + '.nodes')
        arcs_path = os.path.join(trip_path, trip_id + '.arcs')
        route_path = os.path.join(trip_path, trip_id + '.route')

        if not all(os.path.exists(p) for p in [track_path, nodes_path, arcs_path, route_path]):
            continue

        rn = RoadNetwork.from_files(nodes_path, arcs_path)
        track = np.loadtxt(track_path)
        route = [int(line.strip()) for line in open(route_path)]

        gt_arcs = project_track_to_route(track, rn, route)
        cands_per_step, feats_per_step = extract_trip_candidates(rn, track, radius_meters=60.0)

        for t in range(len(track)):
            cands = cands_per_step[t]
            feats = feats_per_step[t]
            true_arc = gt_arcs[t]

            for i, cand_arc in enumerate(cands):
                all_X.append(feats[i])
                all_y.append(1 if cand_arc == true_arc else 0)
                all_groups.append(trip_id)

    X = np.array(all_X)
    y = np.array(all_y)
    groups = np.array(all_groups)

    print('Feature matrix extracted: X shape =', X.shape, 'y shape =', y.shape)
    print('Positive samples =', np.sum(y == 1), 'Negative samples =', np.sum(y == 0))
    print('Total groups =', len(np.unique(groups)))

    print('')
    print('-' * 50)
    print('STARTING BAYESIAN HYPERPARAMETER OPTIMISATION (Optuna)')
    print('-' * 50)

    opt = BayesianOptimizer(
        X=X,
        y=y,
        groups=groups,
        n_splits=5,
        n_trials=10,
        random_seed=42
    )

    best_params, best_score = opt.run_optimization()
    opt.save_artifacts(
        trials_csv_path=os.path.join(base_dir, 'experiments/optimization_trials.csv'),
        best_json_path=os.path.join(base_dir, 'artifacts/configs/best_hyperparameters.json')
    )

    print('')
    print('Refitting selected classifier on training data...')
    fit_start = time.time()
    best_classifier = build_candidate_classifier(**best_params)
    best_classifier.fit(X, y)
    fit_duration = time.time() - fit_start
    print('Model refitted in', round(fit_duration, 2), 'seconds.')

    model_save_path = os.path.join(base_dir, 'artifacts/models/best_classifier.joblib')
    joblib.dump(best_classifier, model_save_path)
    print('Saved refitted classifier to', model_save_path)

    print('')
    print('-' * 50)
    print('EVALUATING ON STUDENT-TESTING TRIPS (Classifier vs HMM)')
    print('-' * 50)

    test_trip_folders = sorted(glob.glob(os.path.join(test_dir, '*')))
    print('Found', len(test_trip_folders), 'student testing trips.')

    trips_eval_data = []

    for trip_path in test_trip_folders:
        trip_id = os.path.basename(trip_path)
        track_path = os.path.join(trip_path, trip_id + '.track')
        nodes_path = os.path.join(trip_path, trip_id + '.nodes')
        arcs_path = os.path.join(trip_path, trip_id + '.arcs')
        route_path = os.path.join(trip_path, trip_id + '.route')

        if not all(os.path.exists(p) for p in [track_path, nodes_path, arcs_path, route_path]):
            continue

        rn = RoadNetwork.from_files(nodes_path, arcs_path)
        track = np.loadtxt(track_path)
        route = [int(line.strip()) for line in open(route_path)]
        gt_arcs = project_track_to_route(track, rn, route)

        cands_per_step, feats_per_step = extract_trip_candidates(rn, track, radius_meters=60.0)

        probs_per_step = []
        for feats in feats_per_step:
            if len(feats) == 0:
                probs_per_step.append(np.array([]))
            else:
                prob = best_classifier.predict_proba(feats)[:, 1]
                probs_per_step.append(prob)

        trips_eval_data.append({
            'trip_id': trip_id,
            'road_network': rn,
            'track': track,
            'ground_truth': gt_arcs,
            'candidates': cands_per_step,
            'probabilities': probs_per_step
        })

    df_results = run_decoder_comparison(
        trips_data=trips_eval_data,
        output_csv_path=os.path.join(base_dir, 'experiments/decoder_comparison.csv'),
        plot_path=os.path.join(base_dir, 'visualizations/routes/decoder_comparison.png')
    )

    print('')
    print('=' * 70)
    print('FINAL DECODER COMPARISON SUMMARY RESULTS')
    print('=' * 70)
    print(df_results[[
        'trip_id', 'obs_count',
        'clf_only_accuracy', 'hmm_viterbi_accuracy',
        'clf_only_continuity', 'hmm_viterbi_continuity'
    ]].to_string(index=False))

    print('')
    print('Aggregated Metrics Across All Student-Testing Trips:')
    print('Classifier-Only Mean Top-1 Accuracy: ', round(df_results['clf_only_accuracy'].mean() * 100, 2), '%')
    print('Classifier + HMM Mean Top-1 Accuracy:', round(df_results['hmm_viterbi_accuracy'].mean() * 100, 2), '%')
    print('Classifier-Only Mean Continuity:    ', round(df_results['clf_only_continuity'].mean() * 100, 2), '%')
    print('Classifier + HMM Mean Continuity:   ', round(df_results['hmm_viterbi_continuity'].mean() * 100, 2), '%')
    print('Average Latency (HMM/Viterbi):      ', round(df_results['hmm_viterbi_latency_ms'].mean(), 3), 'ms / observation')
    print('=' * 70)
    print('MEMBER 3 IMPLEMENTATION & EXPERIMENTATION COMPLETE!')


if __name__ == '__main__':
    main()
