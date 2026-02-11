import unittest
from unittest.mock import MagicMock, patch
from src.builder import SequenceBuilder
from src.config import WorkflowSequence

class TestSequenceBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = SequenceBuilder()

    def test_simple_sequence(self):
        graph = {
            "1": {"step_id": "1", "name": "Start", "tool_id": "tool1", "next_steps": ["2"]},
            "2": {"step_id": "2", "name": "End", "tool_id": "tool2", "next_steps": []}
        }
        seq = self.builder.build_sequence("wf1", graph)
        self.assertEqual(seq.steps, ["1", "2"])
        self.assertEqual(seq.branching_steps, [])
        self.assertEqual(seq.missing_next_step, [])

    def test_branching(self):
        graph = {
            "1": {"step_id": "1", "name": "Start", "tool_id": "tool1", "next_steps": ["2", "3"]},
            "2": {"step_id": "2", "name": "BranchA", "tool_id": "tool2", "next_steps": []},
            "3": {"step_id": "3", "name": "BranchB", "tool_id": "tool3", "next_steps": []}
        }
        seq = self.builder.build_sequence("wf2", graph)
        self.assertIn("1", seq.branching_steps)
        # Sequence order depends on sorting, likely ["1", "2", "3"] since 2<3
        self.assertEqual(seq.steps, ["1", "2", "3"])

    def test_cycle(self):
        graph = {
            "1": {"step_id": "1", "name": "Start", "tool_id": "tool1", "next_steps": ["2"]},
            "2": {"step_id": "2", "name": "Loop", "tool_id": "tool2", "next_steps": ["1"]}
        }
        # DFS should stop at cycle
        seq = self.builder.build_sequence("wf3", graph)
        self.assertTrue(len(seq.steps) > 0)
        # Should contain 1 and 2, order depends on start node
        self.assertEqual(set(seq.steps), {"1", "2"})
        self.assertTrue(len(seq.cycles_detected) > 0)
        self.assertIn("Cycle involving", seq.cycles_detected[0])

    def test_missing_next_step(self):
        graph = {
            "1": {"step_id": "1", "name": "Start", "tool_id": "tool1", "next_steps": ["99"]}
        }
        seq = self.builder.build_sequence("wf4", graph)
        self.assertEqual(seq.steps, ["1"])
        self.assertEqual(seq.missing_next_step, ["99"])

    def test_step_without_tools(self):
        graph = {
            "1": {"step_id": "1", "name": "NoTool", "tool_id": None, "next_steps": []}
        }
        seq = self.builder.build_sequence("wf5", graph)
        self.assertEqual(seq.steps_without_tools, ["1"])

if __name__ == '__main__':
    unittest.main()
