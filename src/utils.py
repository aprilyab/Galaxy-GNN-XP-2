import logging
import sys
import json
import csv
import torch
import random
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from logging.handlers import RotatingFileHandler
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logger(name: str, log_file: str = "project.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def save_json(data: List[Dict], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_tsv(data: List[Dict], filepath: Path, headers: List[str]):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        for row in data:
            writer.writerow([row.get(h, "") for h in headers])

class Vocabulary:
    def __init__(self, special_tokens: List[str] = ["<PAD>", "<UNK>"]):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        for token in special_tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token
        return self.stoi[token]

    def build_from_sequences(self, sequences: List[List[str]]):
        for seq in sequences:
            for tool in seq: self.add_token(tool)

    def encode(self, seq: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in seq]

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
        v = cls()
        v.stoi = data["stoi"]
        v.itos = {int(k): val for k, val in data["itos"].items()}
        return v

class SequenceDataset:
    def __init__(
        self, 
        sequences: List[List[str]], 
        vocab: Vocabulary, 
        context_len: int = 5,
        negative_candidates: Optional[Dict[int, List[int]]] = None,
        num_negatives: int = 5
    ):
        self.X, self.y = [], []
        self.negatives = []
        pad_idx = vocab.stoi["<PAD>"]
        
        for seq in sequences:
            ids = vocab.encode(seq)
            for i in range(1, len(ids)):
                target = ids[i]
                ctx = ids[:i]
                
                # Determine current tool for negative sampling
                current_tool_idx = ctx[-1]
                
                if len(ctx) < context_len:
                    ctx = [pad_idx] * (context_len - len(ctx)) + ctx
                else:
                    ctx = ctx[-context_len:]
                
                self.X.append(torch.tensor(ctx, dtype=torch.long))
                self.y.append(torch.tensor(target, dtype=torch.long))
                
                if negative_candidates and current_tool_idx in negative_candidates:
                    cands = negative_candidates[current_tool_idx]
                    if cands:
                        negs = random.choices(cands, k=num_negatives)
                        self.negatives.append(torch.tensor(negs, dtype=torch.long))
                    else:
                        # Fallback: random from vocab excluding target
                        all_ids = list(vocab.stoi.values())
                        negs = [random.choice(all_ids) for _ in range(num_negatives)]
                        self.negatives.append(torch.tensor(negs, dtype=torch.long))
                else:
                     # No candidates provided or tool not in map
                     self.negatives.append(torch.zeros(num_negatives, dtype=torch.long))

def split_workflows(sequences: List[List[str]], test_size: float = 0.1, val_size: float = 0.1):
    unique = [list(s) for s in sorted(list(set(tuple(x) for x in sequences)))]
    random.seed(42)
    random.shuffle(unique)
    n = len(unique)
    te, va = int(n * test_size), int(n * val_size)
    return unique[te+va:], unique[te:te+va], unique[:te]



def build_transition_graph(sequences: List[List[str]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] and seq[i+1]:
                graph.add_edge(seq[i], seq[i+1])
    return graph

def get_negative_candidates(
    graph: nx.DiGraph, 
    vocab: Vocabulary, 
    exclude_special: bool = True
) -> Dict[int, List[int]]:
    candidates = {}
    all_tokens = set(vocab.stoi.values())
    
    special_indices = {vocab.stoi[t] for t in ["<PAD>", "<UNK>", "<INPUT_DATA>"] if t in vocab.stoi}
    
    for token, idx in vocab.stoi.items():
        if exclude_special and idx in special_indices:
            candidates[idx] = []
            continue
            
        successors = set()
        if graph.has_node(token):
            successors = {vocab.stoi[n] for n in graph.successors(token) if n in vocab.stoi}
            
        # Candidates = All - (Successors + Self + Special)
        valid = list(all_tokens - successors - {idx} - (special_indices if exclude_special else set()))
        if not valid:
             # Fallback if everything is connected (unlikely): sample from all except self
             valid = list(all_tokens - {idx})
        
        candidates[idx] = valid
        
    return candidates
