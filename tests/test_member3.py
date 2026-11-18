"""
Comprehensive unit tests for Member 3: HMM and Viterbi Decoding.
All edge cases tested using synthetic test fixtures.
"""

import pytest
import numpy as np
from src.decoder.hmm import RoadNetwork, MapMatchingHMM
from src.decoder.viterbi import ViterbiDecoder


@pytest.fixture
def sample_road_network():
    """
    Synthetic 5-node directed network:
    Node 0 (0.0, 0.0) -> Node 1 (0.001, 0.0) via Arc 0
    Node 1 (0.001, 0.0) -> Node 2 (0.002, 0.0) via Arc 1 (connected to Arc 0)
    Node 2 (0.002, 0.0) -> Node 3 (0.003, 0.0) via Arc 2 (connected to Arc 1)
    Node 3 (0.003, 0.0) -> Node 4 (0.004, 0.0) via Arc 3 (connected to Arc 2)
    Isolated Arc 4: Node 4 -> Node 0 (reverse feedback loop)
    """
    nodes = np.array([
        [0.0, 0.0],
        [0.001, 0.0],
        [0.002, 0.0],
        [0.003, 0.0],
        [0.004, 0.0]
    ])
    arcs = np.array([
        [0, 1],  # Arc 0
        [1, 2],  # Arc 1
        [2, 3],  # Arc 2
        [3, 4],  # Arc 3
        [4, 0]   # Arc 4
    ])
    return RoadNetwork(nodes, arcs)


def test_valid_connected_arc_transitions(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    # Arc 0 to Arc 1 is a direct 1-hop connection
    prob_direct = hmm.compute_transition_log_prob(prev_arc=0, curr_arc=1, delta_d_gps=111.0, delta_t_sec=1.0)
    assert prob_direct > hmm.disconnected_penalty
    assert np.isfinite(prob_direct)


def test_disconnected_arc_transitions(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    # Arc 2 (2->3) cannot directly jump to Arc 0 (0->1) backwards without traversing 3->4->0
    # But backwards 2 to 0 has no direct or short multi-hop
    prob_disc = hmm.compute_transition_log_prob(prev_arc=2, curr_arc=0, delta_d_gps=50.0, delta_t_sec=1.0)
    # Even if multi-hop exists via 4->0, the distance mismatch is heavily penalized
    assert prob_disc < hmm.direct_transition_log_prob


def test_multihop_road_connectivity(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    # Arc 0 (0->1) to Arc 2 (2->3) skips Arc 1 (1->2), requiring 2 hops
    dist = hmm.get_network_distance(from_arc=0, to_arc=2)
    assert dist is not None
    assert dist > 0.0
    # Multi-hop transition should be valid and finite
    prob_multihop = hmm.compute_transition_log_prob(prev_arc=0, curr_arc=2, delta_d_gps=222.0, delta_t_sec=2.0)
    assert prob_multihop > hmm.disconnected_penalty
    assert np.isfinite(prob_multihop)


def test_missing_candidate_states(sample_road_network):
    """
    When true road segment is missing from candidate set at t=1,
    the Viterbi decoder should still find a coherent path using fallback / multi-hop.
    """
    hmm = MapMatchingHMM(sample_road_network)
    decoder = ViterbiDecoder(hmm)

    gps_track = np.array([
        [0.0005, 0.0, 1.0],
        [0.0015, 0.0, 2.0],  # true arc 1 missing from candidate set
        [0.0025, 0.0, 3.0]
    ])

    candidates = [
        [0],
        [4],  # arc 1 is missing, only arc 4 available
        [2]
    ]
    scores = [
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0])
    ]

    path = decoder.decode(gps_track, candidates, scores)
    assert len(path) == 3
    assert path[0] == 0
    assert path[2] == 2


def test_zero_or_very_small_probabilities(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    cands = [0, 1]
    scores = np.array([0.0, 1e-15])
    log_probs = hmm.compute_emission_log_probs(cands, scores)
    assert np.all(np.isfinite(log_probs))
    assert not np.any(np.isnan(log_probs))


def test_large_gps_gaps(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network, gap_threshold_sec=10.0)
    # Temporal gap of 25 seconds
    prob_gap = hmm.compute_transition_log_prob(prev_arc=1, curr_arc=0, delta_d_gps=500.0, delta_t_sec=25.0)
    assert prob_gap == -2.0  # Decoupled uniform probability, avoids crash


def test_viterbi_backtracking(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    decoder = ViterbiDecoder(hmm)

    gps_track = np.array([
        [0.0005, 0.0, 1.0],
        [0.0015, 0.0, 2.0],
        [0.0025, 0.0, 3.0]
    ])

    candidates = [[0, 4], [1, 4], [2, 4]]
    # Highly noisy emissions where isolated Arc 4 has temporarily higher emission
    scores = [
        np.array([0.8, 0.2]),
        np.array([0.3, 0.7]),  # False spike on Arc 4
        np.array([0.9, 0.1])
    ]

    path = decoder.decode(gps_track, candidates, scores)
    # HMM topological transition should override the single false spike on Arc 4!
    assert path == [0, 1, 2]


def test_correct_sequence_length(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    decoder = ViterbiDecoder(hmm)

    T = 50
    gps_track = np.zeros((T, 3))
    candidates = [[0, 1] for _ in range(T)]
    scores = [np.array([0.5, 0.5]) for _ in range(T)]

    path = decoder.decode(gps_track, candidates, scores)
    assert len(path) == T


def test_empty_candidate_sets(sample_road_network):
    hmm = MapMatchingHMM(sample_road_network)
    decoder = ViterbiDecoder(hmm)

    gps_track = np.array([
        [0.0005, 0.0, 1.0],
        [0.0015, 0.0, 2.0],
        [0.0025, 0.0, 3.0]
    ])

    candidates = [[0], [], [1]]
    scores = [np.array([1.0]), np.array([]), np.array([1.0])]

    path = decoder.decode(gps_track, candidates, scores)
    assert len(path) == 3
    assert path[0] == 0
    assert path[1] == 0  # Replaced gracefully
    assert path[2] == 1
