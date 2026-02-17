import torch
import pandas as pd
from pathlib import Path
from typing import List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.sequence_generation import process_all_connections
from src.utils import setup_logger, Vocabulary, SequenceDataset, split_workflows, build_transition_graph, get_negative_candidates

logger = setup_logger("preparation", "processing.log")

def main():
    INPUT_TSV = Path("data/workflow_connections_simulated.tsv")
    OUT_DIR = Path("data/processed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_TSV.exists():
        logger.error(f"Input TSV {INPUT_TSV} not found. Run extract_connections.py first.")
        return

    # 1. Generate sequences from Connections TSV using Kahn's Algorithm
    all_sequences = process_all_connections(INPUT_TSV)
    logger.info(f"Generated {len(all_sequences)} sequences from connections.")

    # 2. Split workflows FIRST to ensure strict isolation
    train_seqs, val_seqs, test_seqs = split_workflows(all_sequences)
    logger.info(f"Split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")

    # 3. Build Vocabulary strictly from TRAINING split
    vocab = Vocabulary()
    vocab.build_from_sequences(train_seqs, min_count=5)
    vocab.save(OUT_DIR / "vocab.json")
    logger.info(f"Vocabulary build complete. Size: {len(vocab.stoi)}")

    # 4. Build Transition Graph strictly from TRAINING split for negative sampling
    train_graph = build_transition_graph(train_seqs)
    neg_candidates = get_negative_candidates(train_graph, vocab)
    logger.info("Transition graph and negative candidates built from training data.")

    # 5. Process each split and save CSVs
    for name, split_seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        if not split_seqs:
            continue
        
        logger.info(f"Processing {name} dataset...")
        # Use training negative candidates for all splits (representing 'impossible' connections in the known world)
        ds = SequenceDataset(split_seqs, vocab, negative_candidates=neg_candidates)
        
        if ds.X:
            # Context window data
            X_data = [x.tolist() for x in ds.X]
            X_df = pd.DataFrame(X_data, columns=[f'context_{i}' for i in range(len(X_data[0]))])
            X_df.to_csv(OUT_DIR / f"X_{name}.csv", index=False)
            
            # Target tool data
            y_data = [y.item() for y in ds.y]
            y_df = pd.DataFrame(y_data, columns=['target'])
            y_df.to_csv(OUT_DIR / f"y_{name}.csv", index=False)
            
            # Negative samples
            if ds.negatives:
                neg_data = [neg.tolist() for neg in ds.negatives]
                neg_df = pd.DataFrame(neg_data, columns=[f'neg_{i}' for i in range(len(neg_data[0]))])
                neg_df.to_csv(OUT_DIR / f"{name}_negatives.csv", index=False)
            
            logger.info(f"Saved {name} CSV files (Total context windows: {len(ds.X)}).")

if __name__ == "__main__":
    main()
