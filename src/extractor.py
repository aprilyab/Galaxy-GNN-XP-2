from typing import List, Dict, Any, Generator, Tuple
from neo4j import Session
from src.logger import logger

class Neo4jExtractor:
    def __init__(self, session: Session):
        self.session = session

    def fetch_workflow_ids(self, limit: int = 1000, skip: int = 0) -> List[str]:
        """Fetches a batch of workflow IDs."""
        query = """
        MATCH (w:Workflow)
        RETURN w.workflow_id as workflow_id
        ORDER BY w.workflow_id
        SKIP $skip LIMIT $limit
        """
        result = self.session.run(query, skip=skip, limit=limit)
        return [record["workflow_id"] for record in result]

    def fetch_batch_workflow_data(self, workflow_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches steps for a batch of workflows in a single query to avoid N+1 issues.
        Returns a nested dict: {workflow_id: {step_id: step_data}}
        """
        query = """
        MATCH (w:Workflow)-[:HAS_STEP]->(s:Step)
        WHERE w.workflow_id IN $workflow_ids
        OPTIONAL MATCH (s)-[:NEXT_STEP]->(next:Step)
        OPTIONAL MATCH (s)-[:STEP_USES_TOOL]->(t:Tool)
        RETURN 
            w.workflow_id as workflow_id,
            s.step_id as step_id, 
            s.name as step_name, 
            t.tool_id as tool_id,
            collect(next.step_id) as next_step_ids
        """
        result = self.session.run(query, workflow_ids=workflow_ids)
        
        workflows_data = {wid: {} for wid in workflow_ids}
        
        for record in result:
            wid = record["workflow_id"]
            step_id = record["step_id"]
            
            # Basic validation for step_id uniqueness within workflow could be done here 
            # or implicitly handled by dict overwrite (last wins). 
            # We'll log if we see it twice? 
            # For performance, we assume standard overwrite but could check.
            
            workflows_data[wid][step_id] = {
                "step_id": step_id,
                "name": record["step_name"],
                "tool_id": record["tool_id"],
                "next_steps": record["next_step_ids"]
            }
            
        return workflows_data

    def extract_workflows_batch(self, batch_size: int = 100) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Yields workflow data (id, steps_dict) in batches using pagination.
        Optimized to fetch step data for the entire batch at once.
        """
        skip = 0
        while True:
            # 1. Get Batch of IDs
            workflow_ids = self.fetch_workflow_ids(limit=batch_size, skip=skip)
            if not workflow_ids:
                break
            
            # 2. Fetch Data for Batch
            batch_data = self.fetch_batch_workflow_data(workflow_ids)
            
            # 3. Yield results
            for wid in workflow_ids:
                # pass empty dict if no steps found (though unusual)
                yield wid, batch_data.get(wid, {})
            
            skip += batch_size
            logger.info(f"Processed batch starting at offset {skip}")
