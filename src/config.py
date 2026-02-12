import os
from typing import List, Optional,Dict
from pydantic import BaseModel, Field

class NodeSpec(BaseModel):
    name: str
    label: str
    file: str
    id_fields: List[str]
    id_property: str
    prop_fields: List[str] = []

class RelationshipSpec(BaseModel):
    type: str
    file: str
    from_: str = Field(..., alias="from_")
    to: str
    from_id_fields: List[str]
    to_id_fields: List[str]
    prop_fields: List[str] = []
    set_source_target: bool = False

class LoaderConfig(BaseModel):
    nodes: List[NodeSpec]
    relationships: List[RelationshipSpec]

class Neo4jConfig(BaseModel):
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

# Workflow specific output models
class StepMetadata(BaseModel):
    step_id: str
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_version: Optional[str] = None
    next_steps: List[str] = []

class WorkflowSequence(BaseModel):
    workflow_id: str
    steps: List[str]
    steps_metadata: Dict[str, StepMetadata] = {} # Map of step_id to details
    branching_steps: List[str] = []
    missing_next_step: List[str] = []
    steps_without_tools: List[str] = []
    cycles_detected: List[str] = []  # Added cycle tracking

class ExtractionMetrics(BaseModel):
    total_workflows: int = 0
    workflows_missing_next_step: int = 0
    steps_without_tools: int = 0
    duplicate_step_ids: int = 0
    branching_steps: int = 0
    cycles_detected: int = 0 # Added cycle metric
