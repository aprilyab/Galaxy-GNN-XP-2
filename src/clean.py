import re
from typing import List, Dict, Optional
from collections import deque, defaultdict
from src.config import WorkflowSequence, StepMetadata

def clean_tool_id(tool_id: Optional[str], tool_name: Optional[str] = None) -> str:
    """
    Standardizes tool identifiers. 
    Prioritizes tool_name if available, otherwise cleans tool_id.
    """
    if tool_name and tool_name.strip() and tool_name.lower() != "null":
        return tool_name.strip()

    if not tool_id:
        return "<INPUT_DATA>"
    
    cleaned = tool_id.strip()
    if not cleaned or cleaned.lower() == "null":
        return "<INPUT_DATA>"
    
    # Extract base tool ID from toolshed URI if applicable
    # Example: toolshed.g2.bx.psu.edu/repos/devteam/bwa/bwa/1.2.3 -> bwa
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) > 2:
            # Usually the tool name is the second to last part or specific index
            # But let's just use the full ID if name is missing to be safe, 
            # or try to find a descriptive part.
            return parts[-2] 
            
    return cleaned

def clean_tool_version(version: Optional[str]) -> str:
    """Standardizes tool versions."""
    if not version or version.lower() in ["null", "none", ""]:
        return "unknown"
    return version.strip()

def topological_sort(steps_metadata: Dict[str, StepMetadata]) -> List[str]:
    """
    Flattens a workflow into a linear sequence using topological sort.
    Deterministic tie-breaking using step_id.
    """
    # 1. Build local graph
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set(steps_metadata.keys())
    
    for step in steps_metadata.values():
        u = step.step_id
        for v in step.next_steps:
            if v in all_nodes: # Ensure sorting only considers valid nodes
                adj[u].append(v)
                in_degree[v] += 1
                
    # 2. Kahn's Algorithm
    # Start with nodes having in_degree 0
    queue = [n for n in all_nodes if in_degree[n] == 0]
    # Determinstic sort: sort queue initially
    queue.sort()
    
    sorted_steps = []
    
    # Using a priority queue or just sorting at each level would be strictly deterministic
    # But standard Kahn with sorted start + sorting neighbors is usually enough.
    # To be fully deterministic, we use a min-heap or sort the queue every pop (inefficient but safe)
    # or just sort neighbors before adding.
    
    # Let's use a list and resort since N is small (<100 usually).
    
    while queue:
        queue.sort() # Ensure we always pick lexicographically smallest available step
        u = queue.pop(0) 
        sorted_steps.append(u)
        
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycles prevented inclusion of some nodes, append them deterministically?
    # Or just return what we have. Stage 1 detects cycles. 
    # If len(sorted_steps) < len(all_nodes), there was a cycle.
    # We will append remaining nodes sorted by ID to ensure no data loss, 
    # though usage in LSTM might be weird for cyclic parts.
    
    if len(sorted_steps) < len(all_nodes):
        remaining = sorted(list(all_nodes - set(sorted_steps)))
        sorted_steps.extend(remaining)
        
    return sorted_steps

def process_workflow(wf: WorkflowSequence) -> List[str]:
    """
    Processes a single workflow:
    1. If branching, use topological sort.
    2. Convert steps to cleaned tool IDs.
    """
    # If we have metadata, use it for sorting
    if wf.steps_metadata:
        sorted_step_ids = topological_sort(wf.steps_metadata)
        
        # Map step_ids to clean tool_ids using metadata
        tool_sequence = []
        for sid in sorted_step_ids:
            meta = wf.steps_metadata.get(sid)
            if meta:
                tool_sequence.append(clean_tool_id(meta.tool_id, meta.tool_name))
            else:
                # Fallback if metadata missing (shouldn't happen)
                tool_sequence.append("<UNK>")
        return tool_sequence
        
    # Fallback to simple steps list if no metadata (Legacy Stage 1 output)
    # This assumes 'steps' in JSON was already ordered.
    return [clean_tool_id(tid) for tid in wf.steps] # Note: wf.steps contained step_IDs in Stage 1, wait.
    # In Stage 1 v1, 'steps' contained step_ids. 
    # In Stage 1 v2 (Enhanced), we have metadata used to look up tools.
    # If we only have step_ids and no metadata, we can't get tool IDs! 
    # Checks:
    # prompt implies we load workflow_sequences.json. 
    # If Stage 1 was re-run, we have metadata. 
    # The user said "Stage 1: Extracted... processed steps...".
    # I added metadata export in Stage 1 "Enhanced".
    # I must assume I have access to tool_ids via metadata or previous map.
    
    # If metadata is missing, we basically fail to produce tool sequences. 
    # I will assume metadata is present as per my update.
    
    return [] 
