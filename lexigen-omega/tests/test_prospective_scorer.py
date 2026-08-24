import unittest

from lexigen_omega import trajectory  # noqa: F401

# The scorer lives under tools so the scientific policy can be run directly in CI.
# Import it by adding the tools directory through a local module loader.
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "omega_task2_scorer",
    Path(__file__).parents[1] / "tools" / "score_task2_prospective_transfer.py",
)
SCORER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCORER)


class ProspectiveScorerTests(unittest.TestCase):
    def prediction(self):
        return {
            "stage": "task2_prospective_exact_sequence_transfer_r1",
            "status": "frozen_before_Task2_outcome",
            "target_task": "target/repo",
            "Task2_outcome_accessed": False,
            "Task2_timing_accessed": False,
            "prospective_ranking": [
                {"candidate_id": "A", "transferred_prior_harmonic_speedup": 1.4},
                {"candidate_id": "B", "transferred_prior_harmonic_speedup": 1.3},
                {"candidate_id": "C", "transferred_prior_harmonic_speedup": 1.2},
                {"candidate_id": "D", "transferred_prior_harmonic_speedup": 1.1},
            ],
        }

    def result(self):
        return {
            "stage": "tasks2_6_preflight_r1",
            "task": 2,
            "instance_id": "target/repo",
            "campaign_credit_eligible": True,
            "status": "completed",
            "candidate_results": [
                {"candidate": "A", "correct": True, "harmonic_speedup": 1.5, "minimum_speedup": 1.2, "error": None},
                {"candidate": "B", "correct": True, "harmonic_speedup": 1.3, "minimum_speedup": 1.1, "error": None},
                {"candidate": "C", "correct": True, "harmonic_speedup": 1.1, "minimum_speedup": 1.0, "error": None},
                {"candidate": "D", "correct": True, "harmonic_speedup": 0.9, "minimum_speedup": 0.8, "error": None},
            ],
        }

    def test_perfect_prospective_order_is_directionally_supported(self):
        score = SCORER.score_prediction(self.prediction(), self.result())
        self.assertEqual(score["pairwise_concordance"], 1.0)
        self.assertAlmostEqual(score["spearman_rank_correlation"], 1.0)
        self.assertTrue(score["top1_hit"])
        self.assertEqual(
            score["status"],
            "sequence_only_transfer_directionally_supported_on_Task2",
        )

    def test_errored_rows_are_excluded_and_can_force_insufficient_evidence(self):
        result = self.result()
        result["candidate_results"][1]["error"] = "install failed"
        score = SCORER.score_prediction(self.prediction(), result)
        self.assertEqual(score["observed_covered_candidates"], 3)
        self.assertEqual(score["status"], "insufficient_clean_covered_evidence")

    def test_infrastructure_failure_is_not_scored_as_negative(self):
        result = self.result()
        result["status"] = "infrastructure_baseline_failure"
        score = SCORER.score_prediction(self.prediction(), result)
        self.assertEqual(score["status"], "insufficient_evidence_infrastructure_failure")


if __name__ == "__main__":
    unittest.main()
