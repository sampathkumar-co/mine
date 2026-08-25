import unittest

from lexigen_omega.trajectory import (
    attach_proposal_provenance,
    build_hindsight_preferences,
    build_mechanism_preferences,
    ingest_v7_gso_preflight_result,
    parse_v7_gso_proposals,
)


class TrajectoryTests(unittest.TestCase):
    def payload(self, eligible=True):
        return {
            "stage": "tasks2_6_preflight_r1",
            "status": "completed",
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
                    "error": None,
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
                    "error": None,
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
                    "error": None,
                },
            ],
        }

    def proposals(self):
        schema = [
            "proposal_id",
            "arm",
            "primitive_sequence",
            "macro_ids",
            "mechanism",
            "source_visible_preconditions",
            "correctness_risk",
            "files_functions",
            "expected_performance_mechanism",
        ]
        return {
            "schema": schema,
            "timing_feedback_used": False,
            "expert_opt_commit_accessed": False,
            "expert_diff_accessed": False,
            "hints_accessed": False,
            "proposals": [
                ["F1", "v7_full", ["REDUCE", "EXECUTE"], ["V7M-004"], "batch bookkeeping once", "batched tokens", "medium", ["repo.py:eval"], "less Python overhead"],
                ["N1", "v7_no_library", ["SPECIALIZE", "EXECUTE"], [], "single item path", "single decode", "medium", ["repo.py:eval"], "less allocation"],
                ["R1", "v7_random_library", ["REPRESENT", "CERTIFY"], ["V7R-002"], "contiguous view", "numeric ids", "medium", ["repo.py"], "less comparison overhead"],
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

    def test_infrastructure_failure_never_becomes_learning_data(self):
        payload = self.payload()
        payload["status"] = "infrastructure_baseline_failure"
        payload["candidate_results"] = []
        self.assertEqual(ingest_v7_gso_preflight_result(payload), ())

    def test_errored_candidate_row_is_not_used_as_scientific_negative(self):
        payload = self.payload()
        payload["candidate_results"][2]["error"] = "install failed"
        observations = ingest_v7_gso_preflight_result(payload)
        self.assertEqual({x.candidate_id for x in observations}, {"F1", "N1"})

    def test_contaminated_diagnostic_task_is_excluded_by_default(self):
        self.assertEqual(ingest_v7_gso_preflight_result(self.payload(False)), ())

    def test_ineligible_observation_cannot_be_used_for_hindsight(self):
        observations = ingest_v7_gso_preflight_result(
            self.payload(False), require_credit_eligible=False
        )
        with self.assertRaisesRegex(ValueError, "ineligible observations"):
            build_hindsight_preferences(observations)

    def test_frozen_provenance_joins_to_evaluator_outcomes(self):
        observations = ingest_v7_gso_preflight_result(self.payload())
        provenance = parse_v7_gso_proposals(self.proposals())
        joined = attach_proposal_provenance(observations, provenance)
        self.assertEqual(joined[0].provenance.primitive_sequence, ("REDUCE", "EXECUTE"))
        self.assertEqual(joined[0].provenance.macro_ids, ("V7M-004",))
        preferences = build_mechanism_preferences(joined)
        self.assertTrue(preferences)
        self.assertEqual(preferences[0].preferred.candidate_id, "F1")
        self.assertEqual(preferences[0].preferred.mechanism, "batch bookkeeping once")

    def test_post_timing_provenance_is_rejected(self):
        payload = self.proposals()
        payload["timing_feedback_used"] = True
        with self.assertRaisesRegex(ValueError, "timing feedback"):
            parse_v7_gso_proposals(payload)


if __name__ == "__main__":
    unittest.main()
