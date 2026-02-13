import torch
from pathlib import Path
from typing import List

from src.sequence_generation import process_workflow
from src.schema_models import WorkflowSequence
from src.utils import setup_logger, save_json, Vocabulary, SequenceDataset, split_workflows, generate_negative_samples, build_transition_graph, get_negative_candidates

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
            torch.save(torch.stack(ds.X), OUT / f"X_{name}.pt")
            torch.save(torch.stack(ds.y), OUT / f"y_{name}.pt")
            
            # Save the new negatives list (stack it)
            if ds.negatives:
                torch.save(torch.stack(ds.negatives), OUT / f"{name}_neg.pt")
            
            logger.info(f"Saved {name} tensors.")

if __name__ == "__main__":
    main()
