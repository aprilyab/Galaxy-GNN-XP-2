import unittest
import torch
from src.utils import Vocabulary, SequenceDataset, split_workflows

class TestUtils(unittest.TestCase):
    def test_vocabulary(self):
        vocab = Vocabulary(special_tokens=["<PAD>", "<UNK>"])
        vocab.add_token("tool1")
        self.assertEqual(vocab.stoi["tool1"], 2)
        self.assertEqual(vocab.itos[2], "tool1")
        
        encoded = vocab.encode(["tool1", "unknown"])
        self.assertEqual(encoded, [2, 1]) # 1 is <UNK>

    def test_sequence_dataset(self):
        vocab = Vocabulary()
        vocab.add_token("A")
        vocab.add_token("B")
        vocab.add_token("C")
        
        sequences = [["A", "B", "C"]]
        ds = SequenceDataset(sequences, vocab, context_len=2)
        # Windows: 
        # 1. target B, context [PAD, A]
        # 2. target C, context [A, B]
        self.assertEqual(len(ds.X), 2)
        
        # Test first window
        self.assertTrue(torch.equal(ds.X[0], torch.tensor([vocab.stoi["<PAD>"], vocab.stoi["A"]])))
        self.assertEqual(ds.y[0].item(), vocab.stoi["B"])

    def test_split_workflows(self):
        seqs = [[str(i)] for i in range(100)]
        train, val, test = split_workflows(seqs, test_size=0.1, val_size=0.1)
        self.assertEqual(len(test), 10)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(train), 80)

if __name__ == "__main__":
    unittest.main()
