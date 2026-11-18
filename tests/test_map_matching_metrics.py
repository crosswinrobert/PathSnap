
import unittest

from src.evaluation.map_matching_metrics import (
    observation_top1_accuracy,
    candidate_miss_rate,
)


class TestMapMatchingMetrics(unittest.TestCase):

    def test_single_candidate(self):
        self.assertEqual(
            observation_top1_accuracy([1], [0.9], [1]),
            1.0
        )

    def test_equal_scores(self):
        # When scores tie, np.argmax selects the first candidate.
        self.assertEqual(
            observation_top1_accuracy(
                [1, 0], [0.5, 0.5], [1, 1]
            ),
            1.0
        )

    def test_candidate_miss(self):
        self.assertEqual(
            candidate_miss_rate(
                [0, 0, 1, 0], [1, 1, 2, 2]
            ),
            0.5
        )

    def test_empty_input(self):
        self.assertEqual(
            observation_top1_accuracy([], [], []),
            0.0
        )
        self.assertEqual(
            candidate_miss_rate([], []),
            0.0
        )


if __name__ == "__main__":
    unittest.main()
