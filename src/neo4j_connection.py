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
        query = """
        MATCH (w:Workflow)
        RETURN w.workflow_id as workflow_id
        ORDER BY w.workflow_id
        SKIP $skip LIMIT $limit
        """
        result = self.session.run(query, skip=skip, limit=limit)
        return [record["workflow_id"] for record in result]

    def fetch_batch_workflow_data(self, workflow_ids: List[str]) -> Dict[str, Dict[str, Any]]:
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
            t.version as tool_version,
            collect(next.step_id) as next_step_ids
        """
        result = self.session.run(query, workflow_ids=workflow_ids)
        workflows_data = {wid: {} for wid in workflow_ids}
        for record in result:
            workflows_data[record["workflow_id"]][record["step_id"]] = {
                "step_id": record["step_id"],
                "name": record["step_name"],
                "tool_id": record["tool_id"],
                "tool_version": record["tool_version"],
                "next_steps": record["next_step_ids"]
            }
        return workflows_data

    def extract_workflows_batch(self, batch_size: int = 100) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        skip = 0
        while True:
            workflow_ids = self.fetch_workflow_ids(limit=batch_size, skip=skip)
            if not workflow_ids:
                break
            batch_data = self.fetch_batch_workflow_data(workflow_ids)
            for wid in workflow_ids:
                yield wid, batch_data.get(wid, {})
            skip += batch_size
            logger.info(f"Extracted batch at offset {skip}")
