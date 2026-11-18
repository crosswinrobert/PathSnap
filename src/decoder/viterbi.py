"""
Log-Space Viterbi Decoding Implementation for Map-Matching.
Member 3 Implementation - PathSnap Project, PES University.

This module provides:
1. Numerically stable log-space Viterbi forward recurrence.
2. Backpointer tracking and backtracking.
3. Handling of missing/empty candidate sets and extreme GPS gaps.
4. Candidate arc sequence reconstruction.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from src.decoder.hmm import MapMatchingHMM, haversine_distance


class ViterbiDecoder:
    """
    Decodes the most likely sequence of hidden road segments given GPS observations.
    """
    def __init__(self, hmm: MapMatchingHMM):
        self.hmm = hmm

    def decode(
        self,
        gps_track: np.ndarray,
        candidates_per_step: List[List[int]],
        scores_per_step: List[np.ndarray]
    ) -> List[int]:
        """
        Run log-space Viterbi decoding over the candidate lattices.

        Args:
            gps_track: np.ndarray of shape (T, 3) with columns [lon, lat, timestamp]
            candidates_per_step: Length-T list of candidate arc IDs at each step
            scores_per_step: Length-T list of classifier score arrays for each candidate set

        Returns:
            List[int] of predicted arc IDs of length T.
        """
        T = len(gps_track)
        if T == 0:
            return []

        # Step 1: Pre-process steps with empty candidate sets (fallback to previous/nearest)
        cleaned_candidates = []
        cleaned_scores = []
        last_valid_cand = [0]  # fallback arc if sequence starts empty

        for t in range(T):
            cands = candidates_per_step[t]
            scores = scores_per_step[t]
            if len(cands) == 0:
                # Fallback: duplicate last valid candidate with neutral score
                cleaned_candidates.append(list(last_valid_cand))
                cleaned_scores.append(np.array([1.0]))
            else:
                cleaned_candidates.append(cands)
                cleaned_scores.append(scores)
                last_valid_cand = cands

        # Viterbi tables
        # V[t] will hold log-probabilities for each candidate at step t
        V: List[np.ndarray] = []
        # backpointer[t] will hold the index of the best predecessor candidate in V[t-1]
        backpointer: List[np.ndarray] = []

        # Initialization (t = 0)
        init_emissions = self.hmm.compute_emission_log_probs(
            cleaned_candidates[0],
            cleaned_scores[0]
        )
        # Uniform initial state distribution over candidates + emission
        init_v = -np.log(len(cleaned_candidates[0])) + init_emissions
        V.append(init_v)

        # Dynamic Programming Forward Pass (t = 1 to T - 1)
        for t in range(1, T):
            prev_cands = cleaned_candidates[t - 1]
            curr_cands = cleaned_candidates[t]
            curr_emissions = self.hmm.compute_emission_log_probs(curr_cands, cleaned_scores[t])

            prev_lon, prev_lat, prev_time = gps_track[t - 1]
            curr_lon, curr_lat, curr_time = gps_track[t]

            delta_d = haversine_distance(prev_lon, prev_lat, curr_lon, curr_lat)
            delta_t = max(0.001, curr_time - prev_time)

            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            # Compute transition matrix of shape (K_prev, K_curr)
            trans_log_matrix = np.zeros((K_prev, K_curr), dtype=float)
            for i, prev_arc in enumerate(prev_cands):
                for j, curr_arc in enumerate(curr_cands):
                    trans_log_matrix[i, j] = self.hmm.compute_transition_log_prob(
                        prev_arc=prev_arc,
                        curr_arc=curr_arc,
                        delta_d_gps=delta_d,
                        delta_t_sec=delta_t
                    )

            # Vectorized recurrence:
            # candidate_val[i, j] = V[t-1][i] + trans[i, j]
            # V[t][j] = max_i(candidate_val[i, j]) + curr_emission[j]
            prev_v_expanded = V[t - 1][:, np.newaxis]  # shape (K_prev, 1)
            total_cost_matrix = prev_v_expanded + trans_log_matrix  # shape (K_prev, K_curr)

            best_prev_indices = np.argmax(total_cost_matrix, axis=0)  # shape (K_curr,)
            best_prev_values = np.max(total_cost_matrix, axis=0)  # shape (K_curr,)

            v_t = best_prev_values + curr_emissions

            V.append(v_t)
            backpointer.append(best_prev_indices)

        # Step 3: Backtracking
        best_last_idx = int(np.argmax(V[-1]))
        predicted_path = [cleaned_candidates[-1][best_last_idx]]

        curr_idx = best_last_idx
        for t in range(T - 1, 0, -1):
            curr_idx = int(backpointer[t - 1][curr_idx])
            predicted_path.append(cleaned_candidates[t - 1][curr_idx])

        predicted_path.reverse()
        return predicted_path
