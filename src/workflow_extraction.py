import argparse
from pathlib import Path
from collections import Counter

from src.schema_models import Neo4jConfig, ExtractionMetrics
from src.neo4j_connection import Neo4jConnector, Neo4jExtractor
from src.sequence_generation import SequenceBuilder
from src.utils import setup_logger, save_json, save_tsv

logger = setup_logger("extraction", "extraction.log")

def main():
    import os
    parser = argparse.ArgumentParser(description="Galaxy Sequence Extraction")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="data/workflow_sequences.json")
    parser.add_argument("--output-tsv", type=str, default="data/workflow_sequences.tsv")
    parser.add_argument("--uri", type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", type=str, default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", type=str, default=os.getenv("NEO4J_PASSWORD", "password"))
    args = parser.parse_args()

    config = Neo4jConfig(uri=args.uri, user=args.user, password=args.password)
    db = Neo4jConnector(config)
    builder = SequenceBuilder()
    metrics = ExtractionMetrics()
    all_sequences = []

    try:
        with db.session() as session:
            extractor = Neo4jExtractor(session)
            logger.info("Extracting workflows...")
            for workflow_id, step_data in extractor.extract_workflows_batch(args.batch_size):
                metrics.total_workflows += 1
                seq = builder.build_sequence(workflow_id, step_data)
                
                if seq.missing_next_step: metrics.workflows_missing_next_step += 1
                if seq.branching_steps: metrics.branching_steps += 1
                if seq.steps_without_tools: metrics.steps_without_tools += 1
                if seq.cycles_detected: metrics.cycles_detected += 1
                metrics.duplicate_step_ids += sum(v - 1 for v in Counter(seq.steps).values() if v > 1)
                
                all_sequences.append(seq.model_dump())

    finally:
        if all_sequences:
            save_json(all_sequences, Path(args.output_json))
            headers = ["workflow_id", "steps_count", "branching_count", "cycle_count"]
            data = [{
                "workflow_id": s["workflow_id"],
                "steps_count": len(s["steps"]),
                "branching_count": len(s.get("branching_steps", [])),
                "cycle_count": len(s.get("cycles_detected", []))
            } for s in all_sequences]
            save_tsv(data, Path(args.output_tsv), headers)
            logger.info(f"Saved {len(all_sequences)} sequences.")
        db.close()

if __name__ == "__main__":
    main()
