from neo4j import GraphDatabase, Driver, Session
import logging
from contextlib import contextmanager
from typing import Generator
from src.config import Neo4jConfig

logger = logging.getLogger(__name__)

from time import sleep

class Neo4jConnector:
    def __init__(self, config: Neo4jConfig):
        self._uri = config.uri
        self._user = config.user
        self._password = config.password
        self._driver: Optional[Driver] = None

    def connect(self, retries: int = 3, delay: int = 2):
        """Initializes the Neo4j driver with retry logic."""
        if not self._driver:
            for attempt in range(retries):
                try:
                    self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
                    self.verify_connection()
                    logger.info(f"Connected to Neo4j at {self._uri}")
                    return
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(f"Connection failed (attempt {attempt+1}/{retries}): {e}. Retrying in {delay}s...")
                        sleep(delay)
                    else:
                        logger.error(f"Failed to connect to Neo4j after {retries} attempts: {e}")
                        raise

    def verify_connection(self):
        """Verifies connectivity to the database."""
        if self._driver:
            try:
                self._driver.verify_connectivity()
            except Exception as e:
                logger.error(f"Neo4j connectivity verification failed: {e}")
                raise

    def close(self):
        """Closes the Neo4j driver."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed.")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provides a transactional session context."""
        if not self._driver:
            self.connect()
        
        session = self._driver.session()
        try:
            yield session
        except Exception as e:
            logger.error(f"Error during Neo4j session: {e}")
            raise
        finally:
            session.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
