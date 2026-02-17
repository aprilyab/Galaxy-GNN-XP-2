from typing import List, Dict, Set, Any, Optional, Tuple
from collections import defaultdict, deque
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger

logger = setup_logger("sequence_gen", "extraction.log")

def clean_tool_token(tool_id: str) -> str:
    """Standardize tool tokens: remove version if present and handle special nodes."""
    if tool_id == "<INPUT_DATA>":
        return tool_id
    
    # Use the tool's name if it's a generic step type, otherwise use the repository name
    if "/" in tool_id:
        # toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy1 -> fastqc
        parts = tool_id.split("/")
        if len(parts) >= 2:
            return parts[-2]
    return tool_id

def build_topo_sequence(connections: pd.DataFrame) -> List[str]:
    """Build a tool sequence from workflow connections using Kahn's Topological Sort."""
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    step_to_tool = {}
    nodes = set()

    for _, row in connections.iterrows():
        s_id = str(row['source_step_id'])
        t_id = str(row['target_step_id'])
        s_tool = str(row['source_tool'])
        t_tool = str(row['target_tool'])
        
        # Map step ID to tool token (standardized)
        step_to_tool[s_id] = clean_tool_token(s_tool)
        step_to_tool[t_id] = clean_tool_token(t_tool)
        
        adj[s_id].append(t_id)
        in_degree[t_id] += 1
        nodes.add(s_id)
        nodes.add(t_id)
        in_degree.setdefault(s_id, 0)

    # Kahn's algorithm with deterministic ordering (alphabetical on step_id strings)
    queue = deque(sorted([n for n in nodes if in_degree[n] == 0]))
    topo_step_ids = []
    
    while queue:
        u = queue.popleft()
        topo_step_ids.append(u)
        
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        
        # Maintain order for stability
        if queue:
            sorted_q = sorted(list(queue))
            queue = deque(sorted_q)

    # If cycle detected, fallback to numeric step_id sorting (less ideal but keeps process alive)
    if len(topo_step_ids) < len(nodes):
        logger.warning(f"Cycle detected in workflow. Fallback to sorted step IDs.")
        topo_step_ids = sorted(list(nodes), key=lambda x: int(x) if x.isdigit() else x)

    # Convert sorted step IDs back to tool tokens
    return [step_to_tool[sid] for sid in topo_step_ids]

def process_all_connections(tsv_path: Path) -> List[List[str]]:
    """Load connection TSV and generate sequences for each workflow."""
    if not tsv_path.exists():
        logger.error(f"TSV not found: {tsv_path}")
        return []

    df = pd.read_csv(tsv_path, sep="\t")
    logger.info(f"Loaded {len(df)} connections from {tsv_path}")

    sequences = []
    workflow_groups = df.groupby("workflow_id")
    
    for wf_id, group in workflow_groups:
        seq = build_topo_sequence(group)
        if len(seq) >= 2:
            sequences.append(seq)
            
    logger.info(f"Generated {len(sequences)} valid sequences.")
    return sequences

if __name__ == "__main__":
    # Test on simulated data
    TSV = Path("data/workflow_connections_simulated.tsv")
    seqs = process_all_connections(TSV)
    if seqs:
        print(f"Sample Sequence: {seqs[0][:10]}")
