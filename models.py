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


class ResearchLibrarianObservation(BaseModel):
    current_node_id: Optional[str] = None
    current_title: str = ""
    current_domain: str = ""
    current_content: str = ""
    available_citations: List[str] = []
    discovery_path: List[str] = []
    indexed_count: int = 0
    reward: float = 0.0
    done: bool = False
    task_id: str = ""
    task_description: str = ""


class ResearchLibrarianState(BaseModel):
    task_id: str
    current_node_id: Optional[str]
    discovery_path: List[str]
    steps: int
    max_steps: int
    done: bool
    reward: float