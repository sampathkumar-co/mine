import unittest

from lexigen_omega.trajectory import EvaluationObservation, build_hindsight_preferences


class TiedOutcomeTests(unittest.TestCase):
    def test_candidate_ids_do_not_break_scientific_ties(self):
        observations = tuple(
            EvaluationObservation(
                task_id="repo/task",
                candidate_id=candidate,
                artifact_fingerprint=candidate.lower() * 8,
                arm="v7_full",
                valid=False,
                score=0.0,
                metrics={"minimum_speedup": 0.0},
                credit_eligible=True,
                source="test",
            )
            for candidate in ("F1", "N2", "R6")
        )
        self.assertEqual(build_hindsight_preferences(observations), ())

    def test_equal_valid_scores_also_do_not_create_preferences(self):
        observations = tuple(
            EvaluationObservation(
                task_id="repo/task",
                candidate_id=candidate,
                artifact_fingerprint=candidate.lower() * 8,
                arm="v7_full",
                valid=True,
                score=1.1,
                metrics={"minimum_speedup": 1.0},
                credit_eligible=True,
                source="test",
            )
            for candidate in ("A", "B")
        )
        self.assertEqual(build_hindsight_preferences(observations), ())


if __name__ == "__main__":
    unittest.main()
