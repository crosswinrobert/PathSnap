"""
Hidden Markov Model (HMM) for GPS Map-Matching and Route Reconstruction.
Member 3 Implementation - PathSnap Project, PES University.

This module models:
1. Hidden States: Candidate road segments (arcs) associated with each GPS observation.
2. Emission Probabilities: Derived from classifier predictions.
3. Transition Probabilities: Topological connectivity, network path distance vs GPS displacement, and elapsed time.
4. Edge Case Handling: Disconnected road segments, GPS dropouts, and temporal gaps.
"""

import math
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


class RoadNetwork:
    """
    Directed Road Network Graph representation loaded from .nodes and .arcs.
    Optimized with adjacency indices for sub-millisecond transition evaluations.
    """
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes = nodes
        self.arcs = arcs
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)

        self.arc_lengths = np.zeros(self.num_arcs, dtype=float)
        self.arc_to_nodes: Dict[int, Tuple[int, int]] = {}
        self.node_adj: Dict[int, Dict[int, float]] = {}  # u -> {v: length}

        for arc_id in range(self.num_arcs):
            u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])
            self.arc_to_nodes[arc_id] = (u, v)

            u_lon, u_lat = nodes[u, 0], nodes[u, 1]
            v_lon, v_lat = nodes[v, 0], nodes[v, 1]
            length_m = haversine_distance(u_lon, u_lat, v_lon, v_lat)
            self.arc_lengths[arc_id] = max(length_m, 0.1)

            if u not in self.node_adj:
                self.node_adj[u] = {}
            # Keep minimum length if parallel arcs exist
            if v not in self.node_adj[u] or self.arc_lengths[arc_id] < self.node_adj[u][v]:
                self.node_adj[u][v] = self.arc_lengths[arc_id]

    @classmethod
    def from_files(cls, nodes_path: str, arcs_path: str) -> "RoadNetwork":
        nodes = np.loadtxt(nodes_path, dtype=float)
        arcs = np.loadtxt(arcs_path, dtype=int)
        return cls(nodes, arcs)

    def get_arc_nodes(self, arc_id: int) -> Tuple[int, int]:
        return self.arc_to_nodes[arc_id]

    def get_arc_length(self, arc_id: int) -> float:
        return self.arc_lengths[arc_id]


class MapMatchingHMM:
    """
    HMM Model for map-matching GPS trajectories to road network arcs.
    """
    def __init__(
        self,
        road_network: RoadNetwork,
        beta: float = 30.0,
        stay_log_prob: float = -0.5,
        direct_transition_log_prob: float = -0.8,
        disconnected_penalty: float = -15.0,
        gap_threshold_sec: float = 15.0,
        epsilon: float = 1e-7
    ):
        self.rn = road_network
        self.beta = beta
        self.stay_log_prob = stay_log_prob
        self.direct_transition_log_prob = direct_transition_log_prob
        self.disconnected_penalty = disconnected_penalty
        self.gap_threshold_sec = gap_threshold_sec
        self.epsilon = epsilon
        self._path_cache: Dict[Tuple[int, int], float] = {}

    def compute_emission_log_probs(
        self,
        candidate_arcs: List[int],
        candidate_scores: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        if len(candidate_arcs) == 0:
            return np.array([])

        scores = np.asarray(candidate_scores, dtype=float)

        if np.all(scores >= 0) and np.sum(scores) > 0:
            probs = scores + self.epsilon
            probs = probs / np.sum(probs)
            return np.log(probs)

        scaled = scores / max(temperature, 1e-4)
        max_s = np.max(scaled)
        exp_s = np.exp(scaled - max_s) + self.epsilon
        probs = exp_s / np.sum(exp_s)
        return np.log(probs)

    def get_network_distance(self, from_arc: int, to_arc: int) -> Optional[float]:
        """
        Ultra-fast topological shortest path search (up to 3 hops) via adjacency index.
        """
        if from_arc == to_arc:
            return 0.0

        u_from, v_from = self.rn.get_arc_nodes(from_arc)
        u_to, v_to = self.rn.get_arc_nodes(to_arc)

        # 1-hop direct: end of from_arc is start of to_arc
        if v_from == u_to:
            return 0.0

        cache_key = (v_from, u_to)
        if cache_key in self._path_cache:
            d = self._path_cache[cache_key]
            return d if d != float("inf") else None

        adj = self.rn.node_adj

        # 2-hop check: v_from -> u_to directly in adj[v_from]
        v_neighbors = adj.get(v_from, {})
        if u_to in v_neighbors:
            d = v_neighbors[u_to]
            self._path_cache[cache_key] = d
            return d

        # 3-hop check: v_from -> intermediate w -> u_to
        best_d = float("inf")
        for w, len1 in v_neighbors.items():
            w_neighbors = adj.get(w, {})
            if u_to in w_neighbors:
                total_d = len1 + w_neighbors[u_to]
                if total_d < best_d:
                    best_d = total_d

        if best_d != float("inf"):
            self._path_cache[cache_key] = best_d
            return best_d

        self._path_cache[cache_key] = float("inf")
        return None

    def compute_transition_log_prob(
        self,
        prev_arc: int,
        curr_arc: int,
        delta_d_gps: float,
        delta_t_sec: float
    ) -> float:
        if delta_t_sec > self.gap_threshold_sec:
            return -2.0

        if prev_arc == curr_arc:
            seg_len = self.rn.get_arc_length(prev_arc)
            diff = max(0.0, delta_d_gps - seg_len)
            diff_penalty = - (diff / max(self.beta, 1.0))
            return self.stay_log_prob + diff_penalty

        u_from, v_from = self.rn.get_arc_nodes(prev_arc)
        u_to, v_to = self.rn.get_arc_nodes(curr_arc)

        if v_from == u_to:
            target_len = self.rn.get_arc_length(curr_arc)
            diff = abs(target_len - delta_d_gps)
            return self.direct_transition_log_prob - (diff / self.beta)

        net_dist = self.get_network_distance(prev_arc, curr_arc)
        if net_dist is not None:
            total_net_dist = net_dist + self.rn.get_arc_length(curr_arc)
            diff = abs(total_net_dist - delta_d_gps)
            multihop_penalty = -1.5
            return multihop_penalty - (diff / self.beta)

        return self.disconnected_penalty
