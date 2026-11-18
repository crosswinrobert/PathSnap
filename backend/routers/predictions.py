import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.metrics import accuracy_score, f1_score, log_loss

from backend.schemas.response import (
    PredictionResponse,
    RouteEvaluationResult,
)
from src.data_io import load_trip, list_trips
from src.config import load_cfg
from src.features import build_trip_tables, FEATURE_COLS
from src.decoder.hmm import MapMatchingHMM, RoadNetwork
from src.decoder.viterbi import ViterbiDecoder
from src.viz import make_trip_map
from src.evaluation.compare_decoders import evaluate_route_continuity
from experiments.member3_experiments import project_track_to_route

router = APIRouter(prefix="/predict", tags=["Prediction"])

MODEL_PATH = Path("artifacts/models/best_classifier.joblib")
CONFIG_PATH = "artifacts/configs/feature_config.json"
MAP_DIR = Path("backend/generated_maps")

_model = None
_cfg = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        _model = joblib.load(MODEL_PATH)

        if _model.n_features_in_ != len(FEATURE_COLS):
            raise ValueError("Model feature count does not match pipeline.")

        if 1 not in _model.classes_:
            raise ValueError("Model does not contain positive class 1.")

    return _model


def get_config():
    global _cfg

    if _cfg is None:
        _cfg = load_cfg(CONFIG_PATH)

    return _cfg


def evaluate_classifier(trip_id, model):
    """Evaluate candidate classification using server-side labels."""
    eval_trip = load_trip(
        partition="student_test",
        trip_id=trip_id,
        load_route=True
    )

    eval_candidates, _, _ = build_trip_tables(
        eval_trip,
        get_config(),
        with_labels=True
    )

    valid = eval_candidates["label_valid"].astype(bool).to_numpy()

    if not valid.any():
        raise ValueError("No valid reference labels for classifier evaluation.")

    X_eval = eval_candidates.loc[valid, FEATURE_COLS].to_numpy(
        dtype=np.float32
    )
    y_true = eval_candidates.loc[valid, "y"].astype(int).to_numpy()

    if not np.isfinite(X_eval).all():
        raise ValueError("Evaluation features contain NaN or infinity.")

    y_pred = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=model.classes_)
        )
    }


def decode_route(trip, cand_df, point_df, model):
    """Predict a route without loading or using reference labels."""
    X = cand_df[FEATURE_COLS].to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError("Input features contain NaN or infinity.")

    probabilities = model.predict_proba(X)
    positive_class = list(model.classes_).index(1)
    scores = probabilities[:, positive_class]

    if not np.isfinite(scores).all():
        raise ValueError("Classifier produced invalid probabilities.")

    scored = cand_df[["obs_idx", "arc_id"]].copy()
    scored["score"] = scores

    grouped = {
        int(obs_idx): group
        for obs_idx, group in scored.groupby("obs_idx", sort=False)
    }

    ordered_points = point_df.sort_values("obs_idx")

    candidates_per_step = []
    scores_per_step = []

    for obs_idx in ordered_points["obs_idx"]:
        group = grouped.get(int(obs_idx))

        if group is None or group.empty:
            raise ValueError(
                f"No candidates found for GPS observation {obs_idx}."
            )

        candidates_per_step.append(
            group["arc_id"].astype(int).tolist()
        )
        scores_per_step.append(
            group["score"].to_numpy(dtype=float)
        )

    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    arcs = trip.arcs[["u", "v"]].to_numpy(dtype=np.int64)

    road_network = RoadNetwork(nodes=nodes, arcs=arcs)
    hmm = MapMatchingHMM(road_network)
    decoder = ViterbiDecoder(hmm)

    gps_track = ordered_points[["lon", "lat", "t"]].to_numpy(
        dtype=float
    )

    decode_start = time.perf_counter()

    predicted_arcs = decoder.decode(
        gps_track=gps_track,
        candidates_per_step=candidates_per_step,
        scores_per_step=scores_per_step
    )

    decoder_latency_ms_per_observation = (
        (time.perf_counter() - decode_start)
        / max(len(gps_track), 1)
        * 1000.0
    )

    if len(predicted_arcs) != len(ordered_points):
        raise ValueError(
            "Decoder output length does not match GPS observation count."
        )

    return (
        [int(arc_id) for arc_id in predicted_arcs],
        float(decoder_latency_ms_per_observation)
    )


def evaluate_route(trip_id, predicted_arcs, cand_df, decoder_latency):
    """
    Evaluate the reconstructed route against reference arcs.
    Reference data is loaded only in this server-side function.
    """
    eval_trip = load_trip(
        partition="student_test",
        trip_id=trip_id,
        load_route=True
    )

    if eval_trip.route is None or len(eval_trip.route) == 0:
        raise ValueError("Reference route is unavailable for evaluation.")

    nodes = eval_trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    arcs = eval_trip.arcs[["u", "v"]].to_numpy(dtype=np.int64)

    road_network = RoadNetwork(nodes=nodes, arcs=arcs)

    ordered_track = eval_trip.track[["lon", "lat", "t"]].to_numpy(
        dtype=float
    )

    # Reuse Member 3's reference-route-to-observation alignment.
    ground_truth = project_track_to_route(
        ordered_track,
        road_network,
        eval_trip.route.tolist()
    )

    if len(ground_truth) != len(predicted_arcs):
        raise ValueError(
            "Reference and predicted route lengths do not match."
        )

    if not ground_truth:
        raise ValueError("No aligned reference observations available.")

    # Route-level observation metrics.
    route_accuracy = accuracy_score(ground_truth, predicted_arcs)

    route_macro_f1 = f1_score(
        ground_truth,
        predicted_arcs,
        average="macro",
        zero_division=0
    )

    # Reuse Member 3's graph-connectivity definition.
    continuity = evaluate_route_continuity(
        road_network,
        predicted_arcs
    )

    # Candidate miss rate: reference arc absent from that observation's candidates.
    candidate_groups = {
        int(obs_idx): set(group["arc_id"].astype(int).tolist())
        for obs_idx, group in cand_df.groupby("obs_idx", sort=False)
    }

    misses = 0

    for obs_idx, true_arc in enumerate(ground_truth):
        available = candidate_groups.get(obs_idx, set())

        if int(true_arc) not in available:
            misses += 1

    candidate_miss_rate = misses / len(ground_truth)

    return RouteEvaluationResult(
        accuracy=float(route_accuracy),
        macro_f1=float(route_macro_f1),
        continuity=float(continuity),
        candidate_miss_rate=float(candidate_miss_rate),
        decoder_latency_ms_per_observation=float(decoder_latency)
    )


@router.post("/{trip_id}", response_model=PredictionResponse)
def predict_trip(trip_id: str):
    start = time.perf_counter()

    try:
        available_trips = list_trips("student_test")

        if trip_id not in available_trips:
            raise HTTPException(
                status_code=404,
                detail="Trip not found in student-test dataset."
            )

        # Prediction path: reference route is deliberately not loaded.
        trip = load_trip(
            partition="student_test",
            trip_id=trip_id,
            load_route=False
        )

        cand_df, point_df, meta = build_trip_tables(
            trip,
            get_config(),
            with_labels=False
        )

        if cand_df.empty or point_df.empty:
            raise HTTPException(
                status_code=422,
                detail="No usable GPS observations or candidates."
            )

        model = get_model()

        predicted_arcs, decoder_latency = decode_route(
            trip=trip,
            cand_df=cand_df,
            point_df=point_df,
            model=model
        )

        # Student-facing map contains predictions only.
        MAP_DIR.mkdir(parents=True, exist_ok=True)
        map_path = MAP_DIR / f"{trip_id}.html"

        make_trip_map(
            trip,
            out_html=str(map_path),
            predicted_arcs=predicted_arcs,
            show_network=False
        )

        # Separate server-side evaluation paths.
        classifier_evaluation = evaluate_classifier(trip_id, model)

        route_evaluation = evaluate_route(
            trip_id=trip_id,
            predicted_arcs=predicted_arcs,
            cand_df=cand_df,
            decoder_latency=decoder_latency
        )

        return PredictionResponse(
            trip_id=trip_id,
            status="hmm_viterbi",
            predicted_route=predicted_arcs,
            evaluation=classifier_evaluation,
            route_evaluation=route_evaluation,
            processing_time_seconds=round(
                time.perf_counter() - start, 4
            ),
            map_url=f"/maps/{trip_id}.html"
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
