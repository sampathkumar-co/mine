import unittest

from lexigen_omega.trajectory import (
    build_hindsight_preferences,
    ingest_v7_gso_preflight_result,
)


class TrajectoryTests(unittest.TestCase):
    def payload(self, eligible=True):
        return {
            "stage": "tasks2_6_preflight_r1",
            "task": 2,
            "instance_id": "example/repo-task",
            "campaign_credit_eligible": eligible,
            "candidate_results": [
                {
                    "candidate": "F1",
                    "arm": "v7_full",
                    "correct": True,
                    "harmonic_speedup": 1.35,
                    "minimum_speedup": 1.20,
                    "tests_passed": 4,
                    "test_count": 4,
                    "patch_sha256": "a" * 64,
                },
                {
                    "candidate": "N1",
                    "arm": "v7_no_library",
                    "correct": True,
                    "harmonic_speedup": 1.05,
                    "minimum_speedup": 0.99,
                    "tests_passed": 4,
                    "test_count": 4,
                    "patch_sha256": "b" * 64,
                },
                {
                    "candidate": "R1",
                    "arm": "v7_random_library",
                    "correct": False,
                    "harmonic_speedup": 0.0,
                    "minimum_speedup": 0.0,
                    "tests_passed": 2,
                    "test_count": 4,
                    "patch_sha256": "c" * 64,
                },
            ],
        }

    def test_ingests_real_evaluator_shape_and_builds_preferences(self):
        observations = ingest_v7_gso_preflight_result(self.payload())
        self.assertEqual(len(observations), 3)
        self.assertEqual(observations[0].task_id, "example/repo-task")
        pairs = build_hindsight_preferences(observations)
        self.assertTrue(pairs)
        self.assertEqual(pairs[0].preferred_candidate, "F1")
        self.assertIn(pairs[0].rejected_candidate, {"N1", "R1"})

    def test_contaminated_diagnostic_task_is_excluded_by_default(self):
        self.assertEqual(ingest_v7_gso_preflight_result(self.payload(False)), ())

    def test_ineligible_observation_cannot_be_used_for_hindsight(self):
        observations = ingest_v7_gso_preflight_result(
            self.payload(False), require_credit_eligible=False
        )
        with self.assertRaisesRegex(ValueError, "ineligible observations"):
            build_hindsight_preferences(observations)


if __name__ == "__main__":
    unittest.main()
