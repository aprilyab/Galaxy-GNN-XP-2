from neo4j import GraphDatabase, Driver, Session
from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Tuple
from time import sleep
from src.schema_models import Neo4jConfig
from src.utils import setup_logger

logger = setup_logger("database", "extraction.log")

class Neo4jConnector:
    def __init__(self, config: Neo4jConfig):
        self._uri = config.uri
        self._user = config.user
        self._password = config.password
        self._driver = None

    def connect(self, retries: int = 3, delay: int = 2):
        if not self._driver:
            for attempt in range(retries):
                try:
                    self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
                    self._driver.verify_connectivity()
                    logger.info(f"Connected to Neo4j at {self._uri}")
                    return
                except Exception as e:
                    if attempt < retries - 1:
                        sleep(delay)
                    else:
                        raise

    def close(self):
        if self._driver:
            self._driver.close()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        if not self._driver:
            self.connect()
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

class Neo4jExtractor:
    def __init__(self, session: Session):
        self.session = session

    def fetch_workflow_ids(self, limit: int = 1000, skip: int = 0) -> List[str]:
        query = "MATCH (w:Workflow) RETURN w.workflow_id as workflow_id ORDER BY w.workflow_id SKIP $skip LIMIT $limit"
        return [r["workflow_id"] for r in self.session.run(query, skip=skip, limit=limit)]

    def fetch_batch_workflow_data(self, workflow_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetches recursive workflow step data using the STEP_FEEDS_INTO relationship."""
        query = """
        MATCH (w:Workflow)-[:HAS_STEP]->(s:Step)
        WHERE w.workflow_id IN $workflow_ids
        OPTIONAL MATCH (s)-[:STEP_FEEDS_INTO]->(next:Step)
        OPTIONAL MATCH (s)-[:STEP_USES_TOOL]->(t:Tool)
        RETURN 
            w.workflow_id as workflow_id,
            s.step_id as step_id, 
            s.name as step_name, 
            t.tool_id as tool_id,
            t.version as tool_version,
            collect(next.step_id) as next_step_ids
        """
        result = self.session.run(query, workflow_ids=workflow_ids)
        workflows_data = {wid: {} for wid in workflow_ids}
        for r in result:
            workflows_data[r["workflow_id"]][r["step_id"]] = {
                "step_id": r["step_id"], "name": r["step_name"], 
                "tool_id": r["tool_id"], "tool_version": r["tool_version"],
                "next_steps": r["next_step_ids"]
            }
        return workflows_data

    def extract_workflows_batch(self, batch_size: int = 100) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        skip = 0
        while True:
            workflow_ids = self.fetch_workflow_ids(limit=batch_size, skip=skip)
            if not workflow_ids: break
            batch_data = self.fetch_batch_workflow_data(workflow_ids)
            for wid in workflow_ids:
                yield wid, batch_data.get(wid, {})
            skip += batch_size
            logger.info(f"Extracted batch at offset {skip}")

    def fetch_tool_connections(self) -> Generator[Dict[str, Any], None, None]:
        """Extracts high-fidelity tool connections for connection-level sequence generation."""
        query = """
        MATCH (w:Workflow)-[:HAS_STEP]->(s1:Step)-[r:STEP_FEEDS_INTO]->(s2:Step)
        WHERE w.workflow_id IS NOT NULL AND s1.step_id IS NOT NULL AND s2.step_id IS NOT NULL
        OPTIONAL MATCH (s1)-[:STEP_USES_TOOL]->(t1:Tool)
        OPTIONAL MATCH (s2)-[:STEP_USES_TOOL]->(t2:Tool)
        RETURN 
            w.workflow_id AS workflow_id, w.workflow_repository AS workflow_name,
            COALESCE(w.created_at, "n/a") AS created_at,
            s1.step_id AS source_step_id,
            COALESCE(t1.tool_id, s1.name, "<INPUT_DATA>") AS source_tool_raw,
            COALESCE(t1.version, "n/a") AS source_tool_version,
            COALESCE(r.from_output_name, "output") AS source_output_name,
            s2.step_id AS target_step_id,
            COALESCE(t2.tool_id, s2.name, "<INPUT_DATA>") AS target_tool_raw,
            COALESCE(t2.version, "n/a") AS target_tool_version,
            COALESCE(r.input_name, "input") AS target_input_name
        ORDER BY workflow_id, source_step_id
        """
        for record in self.session.run(query):
            yield dict(record)
