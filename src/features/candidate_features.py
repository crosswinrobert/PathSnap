"""
Candidate road segment extraction and feature engineering using high-performance cKDTree.
Member 3 Implementation - PathSnap Project, PES University.
"""

import math
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple, Optional
from src.decoder.hmm import RoadNetwork


def point_to_segment_projection(
    p_lon: float, p_lat: float,
    u_lon: float, u_lat: float,
    v_lon: float, v_lat: float
) -> Tuple[float, float, float]:
    """
    Project point p onto segment u-v.
    Returns:
        dist_meters: Distance from p to nearest point on segment
        t: Projection parameter along segment [0.0, 1.0]
        heading_rad: Direction of vector u->v in radians [-pi, pi]
    """
    cos_lat = math.cos(math.radians((u_lat + v_lat) / 2.0))
    m_per_deg_lat = 111132.95
    m_per_deg_lon = 111132.95 * cos_lat

    dx = (v_lon - u_lon) * m_per_deg_lon
    dy = (v_lat - u_lat) * m_per_deg_lat
    seg_len_sq = dx * dx + dy * dy

    px = (p_lon - u_lon) * m_per_deg_lon
    py = (p_lat - u_lat) * m_per_deg_lat
    heading_rad = math.atan2(dy, dx)

    if seg_len_sq < 1e-4:
        dist = math.sqrt(px * px + py * py)
        return dist, 0.0, heading_rad

    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
    proj_x = t * dx
    proj_y = t * dy
    dist = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    return dist, t, heading_rad


def extract_trip_candidates(
    road_network: RoadNetwork,
    track: np.ndarray,
    radius_meters: float = 75.0,
    max_candidates: int = 8
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    Extract candidate road arcs using spatial cKDTree index.
    Features: [dist_meters, heading_diff, t_ratio, arc_length, speed]
    """
    T = len(track)
    nodes = road_network.nodes
    arcs = road_network.arcs

    # Build cKDTree on arc midpoints
    u_coords = nodes[arcs[:, 0]]
    v_coords = nodes[arcs[:, 1]]
    midpoints = (u_coords + v_coords) / 2.0
    tree = cKDTree(midpoints)

    deg_radius = (radius_meters + 40.0) / 111000.0
    all_cand_indices = tree.query_ball_point(track[:, :2], r=deg_radius)

    all_candidates: List[List[int]] = []
    all_features: List[np.ndarray] = []

    for t in range(T):
        p_lon, p_lat, p_time = track[t]

        # Calculate GPS heading and speed
        if t > 0:
            prev_lon, prev_lat, prev_time = track[t - 1]
            dt = max(0.001, p_time - prev_time)
            gps_dx = (p_lon - prev_lon) * 111132.95 * math.cos(math.radians(p_lat))
            gps_dy = (p_lat - prev_lat) * 111132.95
            gps_heading = math.atan2(gps_dy, gps_dx)
            speed = math.sqrt(gps_dx ** 2 + gps_dy ** 2) / dt
        else:
            gps_heading = 0.0
            speed = 0.0

        cand_arcs = all_cand_indices[t]
        step_cands: List[int] = []
        step_feats: List[List[float]] = []

        for arc_id in cand_arcs:
            u, v = road_network.get_arc_nodes(arc_id)
            dist, t_ratio, seg_heading = point_to_segment_projection(
                p_lon, p_lat, nodes[u, 0], nodes[u, 1], nodes[v, 0], nodes[v, 1]
            )

            if dist <= radius_meters:
                h_diff = abs(gps_heading - seg_heading)
                if h_diff > math.pi:
                    h_diff = 2.0 * math.pi - h_diff
                arc_len = road_network.get_arc_length(arc_id)
                step_cands.append(arc_id)
                step_feats.append([dist, h_diff, t_ratio, arc_len, speed])

        # If empty, query the 1 nearest neighbor
        if len(step_cands) == 0:
            _, nearest_arc = tree.query([p_lon, p_lat], k=1)
            u, v = road_network.get_arc_nodes(nearest_arc)
            dist, t_ratio, seg_heading = point_to_segment_projection(
                p_lon, p_lat, nodes[u, 0], nodes[u, 1], nodes[v, 0], nodes[v, 1]
            )
            step_cands.append(nearest_arc)
            step_feats.append([dist, 0.0, t_ratio, road_network.get_arc_length(nearest_arc), speed])

        # Limit to max_candidates sorted by distance
        if len(step_cands) > max_candidates:
            sorted_idx = np.argsort([f[0] for f in step_feats])[:max_candidates]
            step_cands = [step_cands[i] for i in sorted_idx]
            step_feats = [step_feats[i] for i in sorted_idx]

        all_candidates.append(step_cands)
        all_features.append(np.array(step_feats, dtype=float))

    return all_candidates, all_features
