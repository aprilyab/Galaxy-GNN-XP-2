import unittest
from unittest.mock import MagicMock
from src.neo4j_connection import Neo4jExtractor

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.session = MagicMock()
        self.extractor = Neo4jExtractor(self.session)

    def test_fetch_workflow_ids(self):
        mock_result = [{"workflow_id": "wf1"}, {"workflow_id": "wf2"}]
        self.session.run.return_value = mock_result
        
        ids = self.extractor.fetch_workflow_ids(limit=2)
        self.assertEqual(ids, ["wf1", "wf2"])
        self.session.run.assert_called_once()

    def test_fetch_batch_workflow_data(self):
        mock_record = {
            "workflow_id": "wf1",
            "step_id": "s1",
            "step_name": "Step 1",
            "tool_id": "t1",
            "tool_version": "1.0",
            "next_step_ids": ["s2"]
        }
        self.session.run.return_value = [mock_record]
        
        data = self.extractor.fetch_batch_workflow_data(["wf1"])
        self.assertIn("wf1", data)
        self.assertEqual(data["wf1"]["s1"]["tool_id"], "t1")

if __name__ == "__main__":
    unittest.main()
