import unittest
import torch
import networkx as nx
from src.utils import Vocabulary, SequenceDataset, build_transition_graph, get_negative_candidates

class TestNegativeSampling(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary(special_tokens=["<PAD>", "<UNK>", "<INPUT_DATA>"])
        # Add basic tools
        self.tools = ["T1", "T2", "T3", "T4", "T5"]
        for t in self.tools:
            self.vocab.add_token(t)
            
    def test_build_transition_graph(self):
        sequences = [
            ["T1", "T2", "T3"],
            ["T2", "T4"],
            ["T1", "T5"]
        ]
        graph = build_transition_graph(sequences)
        
        self.assertTrue(isinstance(graph, nx.DiGraph))
        self.assertTrue(graph.has_edge("T1", "T2"))
        self.assertTrue(graph.has_edge("T2", "T3"))
        self.assertTrue(graph.has_edge("T2", "T4"))
        self.assertTrue(graph.has_edge("T1", "T5"))
        self.assertFalse(graph.has_edge("T1", "T4")) # No direct link

    def test_get_negative_candidates(self):
        # Graph: T1 -> {T2, T3}, T2 -> {T4}, T4 -> {}
        graph = nx.DiGraph()
        graph.add_edge("T1", "T2")
        graph.add_edge("T1", "T3")
        graph.add_edge("T2", "T4")
        
        candidates = get_negative_candidates(graph, self.vocab)
        
        t1_idx = self.vocab.stoi["T1"]
        t2_idx = self.vocab.stoi["T2"]
        t3_idx = self.vocab.stoi["T3"]
        t4_idx = self.vocab.stoi["T4"]
        t5_idx = self.vocab.stoi["T5"] # Unconnected in graph
        
        # T1 connects to T2, T3. Candidates should be T4, T5 (and special tokens if handled, but usually we just check tools)
        t1_candidates = candidates[t1_idx]
        self.assertNotIn(t2_idx, t1_candidates)
        self.assertNotIn(t3_idx, t1_candidates)
        self.assertNotIn(t1_idx, t1_candidates) # Self
        self.assertIn(t4_idx, t1_candidates)
        self.assertIn(t5_idx, t1_candidates)

    def test_sequence_dataset_negatives(self):
        # Setup specific graph where T1->T2 only.
        # So for context ending in T1, T2 is positive, everything else is negative.
        sequences = [["T1", "T2"]]
        
        graph = nx.DiGraph()
        graph.add_edge("T1", "T2")
        
        candidates = get_negative_candidates(graph, self.vocab)
        
        ds = SequenceDataset(sequences, self.vocab, negative_candidates=candidates, num_negatives=2)
        
        # X shape: (1, context_len)
        # Negatives shape: (1, num_negatives)
        self.assertEqual(len(ds.negatives), 1)
        self.assertEqual(ds.negatives[0].shape, (2,))
        
        # Verify negative samples for the first item
        # Context is [... T1] (padded)
        # Positive is T2
        # Negatives should NOT be T2
        neg_samples = ds.negatives[0]
        t2_idx = self.vocab.stoi["T2"]
        for neg in neg_samples:
            self.assertNotEqual(neg.item(), t2_idx)
            
if __name__ == "__main__":
    unittest.main()
