import json
import csv
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict

from src.config import Neo4jConfig, ExtractionMetrics
from src.db import Neo4jConnector
from src.extractor import Neo4jExtractor
from src.builder import SequenceBuilder
from src.logger import logger

def save_json(data: List[Dict], filepath: Path):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_tsv(data: List[Dict], filepath: Path):
    if not data:
        return
    
    headers = ["workflow_id", "steps_count", "branching_count", "missing_next_count", "no_tool_count", "cycle_count"]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(headers)
        for item in data:
            writer.writerow([
                item["workflow_id"],
                len(item["steps"]),
                len(item.get("branching_steps", [])),
                len(item.get("missing_next_step", [])),
                len(item.get("steps_without_tools", [])),
                len(item.get("cycles_detected", []))
            ])

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Neo4j -> Sequence Extraction")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for Neo4j extraction")
    parser.add_argument("--output-json", type=str, default="workflow_sequences.json", help="Output JSON path")
    parser.add_argument("--output-tsv", type=str, default="workflow_sequences.tsv", help="Output TSV path")
    args = parser.parse_args()

    config = Neo4jConfig()
    db = Neo4jConnector(config)
    builder = SequenceBuilder()
    
    metrics = ExtractionMetrics()
    all_sequences = []

    try:
        with db.session() as session:
            extractor = Neo4jExtractor(session)
            
            logger.info("Starting extraction...")
            
            # Using partial try-block inside loop or just saving what we have if outer fails?
            # Better to try-catch outside but save what we have in finally or catch block.
            
            for workflow_id, step_data in extractor.extract_workflows_batch(args.batch_size):
                try:
                    metrics.total_workflows += 1
                    
                    seq = builder.build_sequence(workflow_id, step_data)
                    
                    # Update metrics
                    if seq.missing_next_step:
                        metrics.workflows_missing_next_step += 1
                    if seq.branching_steps:
                        metrics.branching_steps += 1
                    if seq.steps_without_tools:
                        metrics.steps_without_tools += 1
                    if seq.cycles_detected:
                        metrics.cycles_detected += 1
                    
                    # Check for duplicate steps
                    step_counts = Counter(seq.steps)
                    duplicates_sum = sum(v - 1 for v in step_counts.values() if v > 1)
                    if duplicates_sum > 0:
                       metrics.duplicate_step_ids += duplicates_sum
                    
                    all_sequences.append(seq.model_dump())
                except Exception as e:
                    logger.error(f"Error processing workflow {workflow_id}: {e}")
                    # Continue to next workflow? 
                    # Yes, robust extraction should continue.
                    continue

        logger.info(f"Extraction complete. Processed {metrics.total_workflows} workflows.")
        logger.info(f"Metrics: {metrics.model_dump_json(indent=2)}")

    except Exception as e:
        logger.error(f"Fatal error during extraction loop: {e}", exc_info=True)
        # We will save what we have so far in finally block or here
    finally:
        # Save outputs (partial or full)
        if all_sequences:
            try:
                save_json(all_sequences, Path(args.output_json))
                logger.info(f"Saved JSON to {args.output_json} (Count: {len(all_sequences)})")
                
                save_tsv(all_sequences, Path(args.output_tsv))
                logger.info(f"Saved TSV to {args.output_tsv}")
            except Exception as e:
                 logger.error(f"Failed to save outputs: {e}")
        
        db.close()

if __name__ == "__main__":
    main()
