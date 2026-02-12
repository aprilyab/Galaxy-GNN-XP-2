from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from src.schema_models import WorkflowSequence, StepMetadata
from src.utils import setup_logger

logger = setup_logger("logic", "extraction.log")

class SequenceBuilder:
    def build_sequence(self, workflow_id: str, steps_graph: Dict[str, Any]) -> WorkflowSequence:
        visited = set()
        sequence = []
        branching_steps = set()
        missing_next = set()
        no_tools = set()
        cycles_detected = []

        all_step_ids = set(steps_graph.keys())
        target_step_ids = set()
        for s in steps_graph.values():
            for ns in s["next_steps"]:
                if ns: target_step_ids.add(ns)
            if not s["tool_id"]: no_tools.add(s["step_id"])

        start_nodes = sorted(list(all_step_ids - target_step_ids))
        if not start_nodes and all_step_ids: start_nodes = [min(all_step_ids)]
        
        stack = [(node, set()) for node in reversed(start_nodes)]
        while stack:
            current_id, path = stack.pop()
            if current_id in path:
                cycles_detected.append(f"Cycle: {current_id}")
                continue
            if current_id in visited: continue
            
            visited.add(current_id)
            sequence.append(current_id)
            step_data = steps_graph.get(current_id)
            if not step_data: continue

            next_ids = sorted([n for n in step_data["next_steps"] if n], reverse=True)
            if len(next_ids) > 1: branching_steps.add(current_id)
            
            new_path = path | {current_id}
            for nid in next_ids:
                if nid not in steps_graph: missing_next.add(nid)
                else: stack.append((nid, new_path))

        return WorkflowSequence(
            workflow_id=workflow_id,
            steps=sequence,
            steps_metadata={
                sid: StepMetadata(
                    step_id=sid,
                    tool_id=steps_graph[sid]["tool_id"],
                    tool_name=steps_graph[sid].get("name"),
                    tool_version=steps_graph[sid].get("tool_version"),
                    next_steps=steps_graph[sid]["next_steps"]
                ) for sid in sequence
            },
            branching_steps=list(branching_steps),
            missing_next_step=list(missing_next),
            steps_without_tools=list(no_tools),
            cycles_detected=cycles_detected
        )

def clean_tool_id(tool_id: Optional[str], tool_name: Optional[str] = None) -> str:
    if tool_name and tool_name.strip() and tool_name.lower() != "null":
        return tool_name.strip()
    if not tool_id or tool_id.lower() == "null":
        return "<INPUT_DATA>"
    return tool_id.split("/")[-2] if "/" in tool_id else tool_id

def topological_sort(steps_metadata: Dict[str, StepMetadata]) -> List[str]:
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set(steps_metadata.keys())
    for step in steps_metadata.values():
        for v in step.next_steps:
            if v in all_nodes:
                adj[step.step_id].append(v)
                in_degree[v] += 1
    
    queue = sorted([n for n in all_nodes if in_degree[n] == 0])
    sorted_steps = []
    while queue:
        queue.sort()
        u = queue.pop(0)
        sorted_steps.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0: queue.append(v)
            
    if len(sorted_steps) < len(all_nodes):
        sorted_steps.extend(sorted(list(all_nodes - set(sorted_steps))))
    return sorted_steps

def process_workflow(wf: WorkflowSequence) -> List[str]:
    if wf.steps_metadata:
        sorted_ids = topological_sort(wf.steps_metadata)
        return [clean_tool_id(wf.steps_metadata[sid].tool_id, wf.steps_metadata[sid].tool_name) for sid in sorted_ids]
    return [clean_tool_id(tid) for tid in wf.steps]
