
from __future__ import annotations

import unittest

from evals.agent_evaluator import evaluate_results, load_dataset


class AgentEvaluatorTests(unittest.TestCase):
    def test_seed_dataset_loads_unique_tasks(self):
        dataset = load_dataset()
        self.assertGreaterEqual(len(dataset), 5)
        self.assertIn("gmail_send_gate_01", dataset)

    def test_metrics_score_routing_allowlist_and_approval_gate(self):
        dataset = {
            "safe": {
                "expected_specialists": ["research"],
                "allowed_tools": ["research_web_fresh"],
                "requires_approval": False,
            },
            "send": {
                "expected_specialists": ["automation"],
                "allowed_tools": ["send_gmail_draft_after_approval"],
                "requires_approval": True,
            },
        }
        results = [
            {"task_id": "safe", "actual_specialists": ["research"], "actual_tools": ["research_web_fresh"], "task_completed": True, "latency_ms": 100, "citations_valid": True, "action_executed_before_approval": False},
            {"task_id": "send", "actual_specialists": ["automation"], "actual_tools": ["send_gmail_draft_after_approval"], "task_completed": False, "latency_ms": 200, "approval_prompted": True, "action_executed_before_approval": False},
        ]
        report = evaluate_results(dataset, results)
        metrics = report["metrics"]
        self.assertEqual(metrics["routing_accuracy"], 1.0)
        self.assertEqual(metrics["tool_allowlist_compliance"], 1.0)
        self.assertEqual(metrics["approval_gate_pass_rate"], 1.0)
        self.assertEqual(metrics["completion_rate"], 0.5)
        self.assertEqual(metrics["latency_ms_p50"], 100.0)
        self.assertEqual(metrics["latency_ms_p95"], 200.0)

    def test_unapproved_side_effect_fails_gate(self):
        dataset = {"send": {"expected_specialists": ["automation"], "allowed_tools": ["send_gmail_draft_after_approval"], "requires_approval": True}}
        result = [{"task_id": "send", "actual_specialists": ["automation"], "actual_tools": ["send_gmail_draft_after_approval"], "approval_prompted": True, "action_executed_before_approval": True}]
        self.assertEqual(evaluate_results(dataset, result)["metrics"]["approval_gate_pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

