from typing import List, Dict, Set, Any, Tuple
from src.config import WorkflowSequence
from src.logger import logger

class SequenceBuilder:
    def __init__(self):
        pass

    def build_sequence(self, workflow_id: str, steps_graph: Dict[str, Any]) -> WorkflowSequence:
        """
        Constructs an ordered sequence from the step graph using iterative DFS.
        Handles branching, cycles, and missing steps.
        """
        visited = set()
        sequence = []
        branching_steps = set()
        missing_next = set()
        no_tools = set()
        cycles_detected = []

        all_step_ids = set(steps_graph.keys())
        target_step_ids = set()
        
        # 1. basic properties and connectivity
        for s in steps_graph.values():
            for ns in s["next_steps"]:
                if ns:
                    target_step_ids.add(ns)
            
            if not s["tool_id"]:
                no_tools.add(s["step_id"])

        # 2. Identify start nodes (no incoming local edges)
        start_nodes = list(all_step_ids - target_step_ids)
        if not start_nodes and all_step_ids:
             # Cycle or component disconnected from start. Heuristic: lowest ID.
             start_nodes = [min(all_step_ids)] 
        
        start_nodes.sort() # Determinism

        # 3. Iterative DFS
        # Stack stores (current_id, path_set) 
        # path_set is used for cycle detection in the current branch
        
        # We need to be careful with "visited" vs "path". 
        # Visited means we processed this node and its children (or added to sequence).
        # In a DAG, if we visit a visited node, we don't need to re-process.
        # But if we see a node in the *current path*, it's a cycle.
        
        # To emulate the recursive pre-order:
        # We process node -> add children to stack.
        
        # However, for cycle detection in iterative DFS without recursion stack is slightly trickier.
        # We can store path in the stack elements. 
        # limit path size? If workflows are huge, storing path sets might be memory intensive.
        # Galaxy workflows usually < 100 steps. Sets are fine.

        # We reverse start_nodes so that when we pop, we process in order.
        stack = [(node, set()) for node in reversed(start_nodes)]

        while stack:
            current_id, path = stack.pop()
            
            if current_id in path:
                cycles_detected.append(f"Cycle involving {current_id}")
                continue # Break cycle path
                
            if current_id in visited:
                continue

            visited.add(current_id)
            sequence.append(current_id)
            
            step_data = steps_graph.get(current_id)
            if not step_data:
                continue

            next_ids = [n for n in step_data["next_steps"] if n]
            
            if len(next_ids) > 1:
                branching_steps.add(current_id)

            next_ids.sort(reverse=True) # Reverse so we pop in increasing order
            
            new_path = path | {current_id}
            
            for nid in next_ids:
                if nid not in steps_graph:
                    missing_next.add(nid)
                else:
                    stack.append((nid, new_path))

        return WorkflowSequence(
            workflow_id=workflow_id,
            steps=sequence,
            steps_metadata={
                sid: {
                    "step_id": sid,
                    "tool_id": steps_graph[sid]["tool_id"],
                    "tool_name": steps_graph[sid].get("name"),
                    "tool_version": steps_graph[sid].get("tool_version"),
                    "next_steps": steps_graph[sid]["next_steps"]
                }
                for sid in sequence
            },
            branching_steps=list(branching_steps),
            missing_next_step=list(missing_next),
            steps_without_tools=list(no_tools),
            cycles_detected=cycles_detected
        )
