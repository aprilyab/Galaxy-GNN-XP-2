import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.sequence_generation import process_all_connections
from src.utils import setup_logger, Vocabulary, SequenceDataset, split_workflows, build_transition_graph, get_negative_candidates

logger = setup_logger("preparation", "processing.log")

def calculate_optimal_sequence_length(sequences: List[List[str]], percentile: int = 95) -> int:
    lengths = [len(seq) for seq in sequences if seq]
    if not lengths:
        return 10
    
    optimal_length = int(np.percentile(lengths, percentile))
    max_length = max(lengths)
    
    logger.info(f"Sequence length stats - Min: {min(lengths)}, Max: {max_length}, "
                f"Mean: {np.mean(lengths):.1f}, 95th percentile: {optimal_length}")
    
    return min(optimal_length, max_length)

def augment_sequences(sequences: List[List[str]], augment_factor: float = 1.5) -> List[List[str]]:
    augmented = sequences.copy()
    
    for seq in sequences:
        if len(seq) < 3:
            continue
            
        for length in range(3, len(seq)):
            if np.random.random() < 0.3:
                start_idx = np.random.randint(0, len(seq) - length + 1)
                subsequence = seq[start_idx:start_idx + length]
                augmented.append(subsequence)
    
    logger.info(f"Data augmentation: {len(sequences)} -> {len(augmented)} sequences")
    return augmented

def save_variable_length_csv(seqs: List[List[str]], vocab: Vocabulary, output_path: Path, name: str):
    if not seqs:
        logger.warning(f"No sequences to save for {name}")
        return
    
    encoded_sequences = []
    sequence_lengths = []
    
    for seq in seqs:
        encoded = vocab.encode(seq)
        encoded_sequences.append(encoded)
        sequence_lengths.append(len(encoded))
    
    max_len = max(sequence_lengths) if sequence_lengths else 1
    pad_idx = vocab.stoi["<PAD>"]
    
    padded_sequences = []
    for encoded in encoded_sequences:
        padded = encoded + [pad_idx] * (max_len - len(encoded))
        padded_sequences.append(padded)
    
    df = pd.DataFrame(padded_sequences, columns=[f'pos_{i}' for i in range(max_len)])
    df.to_csv(output_path / f"X_{name}.csv", index=False)
    
    lengths_df = pd.DataFrame(sequence_lengths, columns=['length'])
    lengths_df.to_csv(output_path / f"lengths_{name}.csv", index=False)
    
    logger.info(f"Saved {name}: {len(seqs)} sequences, max length: {max_len}")

def main():
    INPUT_TSV = Path("data/workflow_connections_simulated.tsv")
    OUT_DIR = Path("data/processed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_TSV.exists():
        logger.error(f"Input TSV {INPUT_TSV} not found. Run extract_connections.py first.")
        return

    all_sequences = process_all_connections(INPUT_TSV)
    logger.info(f"Generated {len(all_sequences)} sequences from connections.")

    train_seqs, val_seqs, test_seqs = split_workflows(all_sequences)
    logger.info(f"Split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")

    train_seqs_augmented = augment_sequences(train_seqs, augment_factor=1.5)
    logger.info(f"Training sequences after augmentation: {len(train_seqs_augmented)}")

    vocab = Vocabulary()
    vocab.build_from_sequences(train_seqs_augmented, min_count=3)
    vocab.save(OUT_DIR / "vocab.json")
    logger.info(f"Vocabulary build complete. Size: {len(vocab.stoi)}")

    train_graph = build_transition_graph(train_seqs_augmented)
    neg_candidates = get_negative_candidates(train_graph, vocab)
    logger.info("Transition graph and negative candidates built from training data.")

    optimal_length = calculate_optimal_sequence_length(train_seqs_augmented, percentile=90)
    logger.info(f"Using optimal sequence length: {optimal_length}")

    for name, split_seqs in [("train", train_seqs_augmented), ("val", val_seqs), ("test", test_seqs)]:
        if not split_seqs:
            continue
        
        logger.info(f"Processing {name} dataset...")
        
        ds = SequenceDataset(split_seqs, vocab, context_len=optimal_length, 
                         negative_candidates=neg_candidates, num_negatives=5)
        
        if ds.X:
            X_data = [x.tolist() for x in ds.X]
            X_df = pd.DataFrame(X_data, columns=[f'context_{i}' for i in range(len(X_data[0]))])
            X_df.to_csv(OUT_DIR / f"X_{name}.csv", index=False)
            
            y_data = [y.item() for y in ds.y]
            y_df = pd.DataFrame(y_data, columns=['target'])
            y_df.to_csv(OUT_DIR / f"y_{name}.csv", index=False)
            
            if ds.negatives:
                neg_data = [neg.tolist() for neg in ds.negatives]
                neg_df = pd.DataFrame(neg_data, columns=[f'neg_{i}' for i in range(len(neg_data[0]))])
                neg_df.to_csv(OUT_DIR / f"{name}_negatives.csv", index=False)
            
            logger.info(f"Saved {name} dataset: {len(X_data)} samples, context length: {len(X_data[0])}")

    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
