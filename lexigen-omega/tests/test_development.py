import unittest

from lexigen_omega.development import DevelopmentArchive, DevelopmentObservation


class DevelopmentArchiveTests(unittest.TestCase):
    def test_exposed_development_evidence_cannot_claim_final_credit(self):
        archive = DevelopmentArchive()
        with self.assertRaisesRegex(ValueError, "final-claim"):
            archive.add(
                DevelopmentObservation(
                    task_id="dev-task",
                    family="repo",
                    mechanism_key="seq:REDUCE>EXECUTE",
                    reward=1.0,
                    source_campaign="failed-v7",
                    final_claim_eligible=True,
                )
            )

    def test_unexposed_task_cannot_enter_development_archive(self):
        archive = DevelopmentArchive()
        with self.assertRaisesRegex(ValueError, "exposed tasks only"):
            archive.add(
                DevelopmentObservation(
                    task_id="blind-task",
                    family="repo",
                    mechanism_key="seq:EXECUTE>CERTIFY",
                    reward=0.5,
                    source_campaign="future-blind",
                    exposed=False,
                )
            )

    def test_infrastructure_incident_has_zero_learning_weight(self):
        archive = DevelopmentArchive()
        archive.extend(
            [
                DevelopmentObservation(
                    task_id="task-a",
                    family="repo-a",
                    mechanism_key="seq:REDUCE>EXECUTE",
                    reward=1.0,
                    source_campaign="dev",
                ),
                DevelopmentObservation(
                    task_id="task-infra",
                    family="repo-b",
                    mechanism_key="seq:EXECUTE>CERTIFY",
                    reward=0.0,
                    source_campaign="dev",
                    scientific=False,
                ),
            ]
        )
        ranked = archive.rank_mechanisms(exploration=0.0)
        self.assertEqual([row.mechanism_key for row in ranked], ["seq:REDUCE>EXECUTE"])

    def test_pairwise_volume_is_aggregated_per_task(self):
        archive = DevelopmentArchive()
        for _ in range(20):
            archive.add(
                DevelopmentObservation(
                    task_id="same-task",
                    family="repo",
                    mechanism_key="seq:A>B",
                    reward=1.0,
                    source_campaign="dev",
                )
            )
        archive.add(
            DevelopmentObservation(
                task_id="other-task",
                family="repo2",
                mechanism_key="seq:C>D",
                reward=0.75,
                source_campaign="dev",
            )
        )
        ranked = archive.rank_mechanisms(exploration=0.0)
        by_key = {row.mechanism_key: row for row in ranked}
        self.assertEqual(by_key["seq:A>B"].task_count, 1)
        self.assertEqual(by_key["seq:A>B"].reward_sum, 1.0)
        self.assertEqual(by_key["seq:C>D"].reward_sum, 0.75)


if __name__ == "__main__":
    unittest.main()
