from typing import List, Dict, Set, Any, Optional, Tuple
from collections import defaultdict, deque
import pandas as pd
from pathlib import Path
import sys
import re
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logger

logger = setup_logger("sequence_gen", "extraction.log")

def clean_tool_token(tool_id: str) -> str:
    """Enhanced tool token extraction with semantic meaning preservation."""
    if tool_id == "<INPUT_DATA>":
        return tool_id
    
    # Handle repository-style tool IDs: toolshed.g2.bx.psu.edu/repos/devteam/fastqc/fastqc/0.74+galaxy1
    if "/" in tool_id:
        parts = tool_id.split("/")
        if len(parts) >= 3:
            # Extract meaningful tool name from repository path
            tool_name = parts[-3] if parts[-3] != parts[-2] else parts[-2]
            # Clean up common patterns
            tool_name = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name)
            tool_name = re.sub(r'_+', '_', tool_name).strip('_')
            return tool_name.lower() if tool_name else "unknown_tool"
    
    # Handle hash-like IDs - try to extract semantic meaning
    if re.match(r'^[a-f0-9]{32}$', tool_id):
        return f"tool_{tool_id[:8]}"  # Shorten hash for readability
    
    # Clean up any remaining special characters
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_id))
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    return cleaned.lower() if cleaned else "unknown_tool"

def extract_tool_metadata(tool_id: str, tool_version: str = None) -> Dict[str, Any]:
    """Extract comprehensive metadata from tool information."""
    metadata = {
        'original_id': tool_id,
        'clean_name': clean_tool_token(tool_id),
        'version': tool_version or 'unknown',
        'category': 'unknown',
        'input_types': [],
        'output_types': []
    }
    
    # Extract category from repository path
    if "/" in tool_id:
        parts = tool_id.split("/")
        if len(parts) >= 3:
            repo_owner = parts[-4] if len(parts) >= 4 else 'unknown'
            metadata['category'] = repo_owner.lower()
    
    # Common Galaxy tool categories based on naming patterns
    tool_name = metadata['clean_name'].lower()
    if any(keyword in tool_name for keyword in ['fastqc', 'quality', 'trim']):
        metadata['category'] = 'quality_control'
    elif any(keyword in tool_name for keyword in ['align', 'blast', 'bowtie']):
        metadata['category'] = 'alignment'
    elif any(keyword in tool_name for keyword in ['assemble', 'spades']):
        metadata['category'] = 'assembly'
    elif any(keyword in tool_name for keyword in ['count', 'feature', 'annotation']):
        metadata['category'] = 'analysis'
    
    return metadata

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
        
        # Map step ID to tool token 
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

def build_dataflow_sequence(connections: pd.DataFrame) -> List[str]:
    """Build sequence based on data flow dependencies, grouping parallel operations."""
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    step_to_tool = {}
    nodes = set()
    
    for _, row in connections.iterrows():
        s_id = str(row['source_step_id'])
        t_id = str(row['target_step_id'])
        s_tool = str(row['source_tool'])
        t_tool = str(row['target_tool'])
        
        step_to_tool[s_id] = clean_tool_token(s_tool)
        step_to_tool[t_id] = clean_tool_token(t_tool)
        
        adj[s_id].append(t_id)
        in_degree[t_id] += 1
        nodes.add(s_id)
        nodes.add(t_id)
        in_degree.setdefault(s_id, 0)
    
    # Group by dependency levels (breadth-first approach)
    levels = []
    remaining_nodes = set(nodes)
    
    while remaining_nodes:
        current_level = [n for n in remaining_nodes if in_degree[n] == 0]
        if not current_level:
            # Handle cycles by breaking randomly
            current_level = [list(remaining_nodes)[0]]
        
        # Sort for consistency
        current_level.sort()
        levels.append(current_level)
        
        # Remove processed nodes and update degrees
        for node in current_level:
            remaining_nodes.remove(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
    
    # Convert levels to sequence
    sequence = []
    for level in levels:
        level_tools = [step_to_tool[node] for node in level if node in step_to_tool]
        sequence.extend(level_tools)
    
    return sequence

def build_frequency_sequence(connections: pd.DataFrame) -> List[str]:
    """Build sequence prioritizing frequently used tools."""
    # Count tool frequencies
    tool_counts = defaultdict(int)
    step_to_tool = {}
    
    for _, row in connections.iterrows():
        s_tool = clean_tool_token(str(row['source_tool']))
        t_tool = clean_tool_token(str(row['target_tool']))
        
        step_to_tool[str(row['source_step_id'])] = s_tool
        step_to_tool[str(row['target_step_id'])] = t_tool
        
        tool_counts[s_tool] += 1
        tool_counts[t_tool] += 1
    
    # Get topological order first
    base_sequence = build_topo_sequence(connections)
    
    # Sort by frequency (higher frequency tools appear earlier in their position)
    # This maintains topological constraints while prioritizing common tools
    return sorted(base_sequence, key=lambda x: -tool_counts[x])

def generate_sequence_variants(connections: pd.DataFrame) -> List[List[str]]:
    """Generate multiple sequence representations for robustness."""
    variants = []
    
    try:
        # Primary: Topological sequence
        topo_seq = build_topo_sequence(connections)
        if len(topo_seq) >= 2:
            variants.append(topo_seq)
        
        # Secondary: Dataflow sequence
        dataflow_seq = build_dataflow_sequence(connections)
        if len(dataflow_seq) >= 2 and dataflow_seq != topo_seq:
            variants.append(dataflow_seq)
        
        # Tertiary: Frequency-optimized sequence
        freq_seq = build_frequency_sequence(connections)
        if len(freq_seq) >= 2 and freq_seq != topo_seq and freq_seq != dataflow_seq:
            variants.append(freq_seq)
            
    except Exception as e:
        logger.warning(f"Error generating sequence variants: {e}")
        try:
            fallback_seq = build_topo_sequence(connections)
            if len(fallback_seq) >= 2:
                variants.append(fallback_seq)
        except Exception as fallback_error:
            logger.error(f"Failed to generate any sequence: {fallback_error}")
    
    return variants

def process_all_connections(tsv_path: Path) -> List[List[str]]:
    """Load connection TSV and generate enhanced sequences for each workflow."""
    if not tsv_path.exists():
        logger.error(f"TSV not found: {tsv_path}")
        return []

    df = pd.read_csv(tsv_path, sep="\t")
    logger.info(f"Loaded {len(df)} connections from {tsv_path}")

    sequences = []
    workflow_groups = df.groupby("workflow_id")
    
    for wf_id, group in workflow_groups:
        # Generate multiple variants per workflow
        variants = generate_sequence_variants(group)
        sequences.extend(variants)
            
    logger.info(f"Generated {len(sequences)} total sequences from {len(workflow_groups)} workflows.")
    return sequences

if __name__ == "__main__":
   main()
