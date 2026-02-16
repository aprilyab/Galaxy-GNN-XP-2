import torch
import pandas as pd
from pathlib import Path
from typing import List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.sequence_generation import process_workflow
from src.schema_models import WorkflowSequence
from src.utils import setup_logger, save_json, Vocabulary, SequenceDataset, split_workflows, build_transition_graph, get_negative_candidates

logger = setup_logger("processing", "processing.log")

def load_data(path: str) -> List[WorkflowSequence]:
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return [WorkflowSequence(**item) for item in (data.get("sequences", data) if isinstance(data, dict) else data)]

def main():
    INPUT = "data/workflow_sequences.json"
    OUT = Path("data/processed")
    OUT.mkdir(parents=True, exist_ok=True)
    
    try:
        workflows = load_data(INPUT)
        logger.info(f"Loaded {len(workflows)} workflows.")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        return

    clean_seqs = [process_workflow(wf) for wf in workflows if process_workflow(wf)]
    logger.info(f"Cleaned {len(clean_seqs)} sequences.")

    vocab = Vocabulary()
    vocab.build_from_sequences(clean_seqs)
    vocab.save(OUT / "vocab.json")

    # Build graph and negative candidates
    graph = build_transition_graph(clean_seqs)
    neg_candidates = get_negative_candidates(graph, vocab)
    logger.info("Built transition graph and negative candidates.")

    train_s, val_s, test_s = split_workflows(clean_seqs)
    
    for name, seqs in [("train", train_s), ("val", val_s), ("test", test_s)]:
        if not seqs: continue
        
        ds = SequenceDataset(seqs, vocab, negative_candidates=neg_candidates)
        if ds.X:
            # Convert tensors to lists for CSV
            X_data = [x.tolist() for x in ds.X]
            y_data = [y.item() for y in ds.y]
            
            # Save X as CSV
            X_df = pd.DataFrame(X_data, columns=[f'context_{i}' for i in range(len(X_data[0]))])
            X_df.to_csv(OUT / f"X_{name}.csv", index=False)
            
            # Save y as CSV
            y_df = pd.DataFrame(y_data, columns=['target'])
            y_df.to_csv(OUT / f"y_{name}.csv", index=False)
            
            # Save negatives as CSV if available
            if ds.negatives:
                neg_data = [neg.tolist() for neg in ds.negatives]
                neg_df = pd.DataFrame(neg_data, columns=[f'neg_{i}' for i in range(len(neg_data[0]))])
                neg_df.to_csv(OUT / f"{name}_negatives.csv", index=False)
            
            logger.info(f"Saved {name} CSV files.")

if __name__ == "__main__":
    main()
