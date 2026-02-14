import unittest
from src.sequence_generation import SequenceBuilder, clean_tool_id, topological_sort
from src.schema_models import StepMetadata

class TestLogic(unittest.TestCase):
    def setUp(self):
        self.builder = SequenceBuilder()

    def test_clean_tool_id(self):
        # Toolshed URI
        self.assertEqual(clean_tool_id("toolshed.g2.bx.psu.edu/repos/devteam/bwa/bwa/1.2.3"), "bwa")
        # Direct ID
        self.assertEqual(clean_tool_id("upload1"), "upload1")
        # Tool Name priority
        self.assertEqual(clean_tool_id("upload1", "Input dataset"), "Input dataset")
        # Null handling
        self.assertEqual(clean_tool_id(None), "<INPUT_DATA>")
        self.assertEqual(clean_tool_id("null"), "<INPUT_DATA>")

    def test_topological_sort(self):
        metadata = {
            "0": StepMetadata(step_id="0", next_steps=["1"]),
            "1": StepMetadata(step_id="1", next_steps=["2"]),
            "2": StepMetadata(step_id="2", next_steps=[])
        }
        self.assertEqual(topological_sort(metadata), ["0", "1", "2"])

    def test_topological_sort_branching(self):
        metadata = {
            "A": StepMetadata(step_id="A", next_steps=["B", "C"]),
            "B": StepMetadata(step_id="B", next_steps=["D"]),
            "C": StepMetadata(step_id="C", next_steps=["D"]),
            "D": StepMetadata(step_id="D", next_steps=[])
        }
        order = topological_sort(metadata)
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")
        self.assertTrue(set(order) == {"A", "B", "C", "D"})

    def test_build_sequence_linear(self):
        graph = {
            "1": {"step_id": "1", "tool_id": "T1", "next_steps": ["2"]},
            "2": {"step_id": "2", "tool_id": "T2", "next_steps": []}
        }
        seq = self.builder.build_sequence("wf1", graph)
        self.assertEqual(seq.steps, ["1", "2"])
        self.assertEqual(len(seq.steps_metadata), 2)

    def test_build_sequence_cycle(self):
        graph = {
            "1": {"step_id": "1", "tool_id": "T1", "next_steps": ["2"]},
            "2": {"step_id": "2", "tool_id": "T2", "next_steps": ["1"]}
        }
        seq = self.builder.build_sequence("wf2", graph)
        self.assertTrue(len(seq.cycles_detected) > 0)
        self.assertEqual(set(seq.steps), {"1", "2"})

    def test_process_workflow_filters_input(self):
        from src.sequence_generation import process_workflow
        from src.schema_models import WorkflowSequence, StepMetadata
        
        # Scenario: T1 -> T2, with T1 being an input/null tool
        wf = WorkflowSequence(
            workflow_id="wf3",
            steps=["1", "2"],
            steps_metadata={
                "1": StepMetadata(step_id="1", tool_id=None, next_steps=["2"]),
                "2": StepMetadata(step_id="2", tool_id="T2", next_steps=[])
            }
        )
        # process_workflow should return only ["T2"] because "1" resolves to <INPUT_DATA>
        result = process_workflow(wf)
        self.assertEqual(result, ["T2"])

if __name__ == "__main__":
    unittest.main()
