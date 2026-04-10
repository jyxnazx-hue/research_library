from typing import List, Literal, Optional, Union
from pydantic import BaseModel

class SearchAction(BaseModel):
    action_type: Literal["search"] = "search"
    query: str

class ReadAction(BaseModel):
    action_type: Literal["read"] = "read"
    node_id: str

class SubmitAction(BaseModel):
    action_type: Literal["submit"] = "submit"
    answer: str

ActionType = Union[SearchAction, ReadAction, SubmitAction]

class ResearchNode(BaseModel):
    node_id: str
    title: str
    domain: str
    abstract: str
    content_text: str
    citations: List[str] = []

class LibraryObservation(BaseModel):
    current_node_id: Optional[str] = None
    current_title: str = ""
    current_domain: str = ""
    current_content: str = ""
    available_citations: List[str] = []
    discovery_path: List[str] = []
    thought_log: List[str] = [] # NEW: For "Neural Trace"
    graph_mermaid: str = ""      # NEW: For visual citation mapping
    reward: float = 0.0
    done: bool = False
    task_id: str = ""
    task_description: str = ""
