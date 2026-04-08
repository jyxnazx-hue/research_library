import json
from pathlib import Path
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import (
    ActionType,
    SearchAction,
    ReadAction,
    SubmitAction,
    ResearchLibrarianObservation,
    ResearchLibrarianState,
)

from server.grader import grade_research_discovery


class ResearchLibrarianEnvironment(Environment):
    def __init__(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"

        with open(data_dir / "nodes.json", "r", encoding="utf-8") as f:
            nodes = json.load(f)

        with open(data_dir / "tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)

        self.nodes: Dict[str, dict] = {node["node_id"]: node for node in nodes}
        self.tasks: Dict[str, dict] = {task["task_id"]: task for task in tasks}

        self.task_id: str = ""
        self.current_node_id: Optional[str] = None
        self.discovery_path: List[str] = []
        self.steps: int = 0
        self.max_steps: int = 8
        self.done: bool = False
        self.reward: float = 0.0

    async def reset(self, task_id: Optional[str] = None) -> ResearchLibrarianObservation:
        self.task_id = task_id or "identify_technology"
        self.steps = 0
        self.done = False
        self.reward = 0.0
        self.discovery_path = []

        if self.task_id == "identify_technology":
            self.current_node_id = "QUANTUMPHYS02"
        elif self.task_id == "chemical_ratio":
            self.current_node_id = "MATSCI03"
        elif self.task_id == "final_synthesis":
            self.current_node_id = "CSMATH01"
        else:
            self.current_node_id = "CSMATH01"

        start_node = self.nodes[self.current_node_id]
        self.discovery_path.append(start_node["domain"])

        return self._build_observation()

    async def step(self, action: ActionType) -> ResearchLibrarianObservation:
        if self.done:
            return self._build_observation()

        self.steps += 1
        step_reward = -0.05

        if isinstance(action, SearchAction):
            query = action.query.lower()
            matched = None

            for node in self.nodes.values():
                haystack = " ".join([
                    node.get("title", ""),
                    node.get("abstract", ""),
                    node.get("content_text", ""),
                    node.get("domain", "")
                ]).lower()
                if query in haystack:
                    matched = node
                    break

            if matched:
                self.current_node_id = matched["node_id"]
                if matched["domain"] not in self.discovery_path:
                    self.discovery_path.append(matched["domain"])
                    step_reward += 0.2
                step_reward += 0.1

        elif isinstance(action, ReadAction):
            if action.node_id in self.nodes:
                self.current_node_id = action.node_id
                node = self.nodes[action.node_id]
                if node["domain"] not in self.discovery_path:
                    self.discovery_path.append(node["domain"])
                    step_reward += 0.25
                step_reward += 0.1

        elif isinstance(action, SubmitAction):
            score = grade_research_discovery(self.task_id, action.answer)
            step_reward += score
            self.done = True

        if self.steps >= self.max_steps:
            self.done = True

        self.reward = max(0.0, min(1.0, self.reward + step_reward))
        return self._build_observation()

    @property
    def state(self) -> ResearchLibrarianState:
        return ResearchLibrarianState(
            task_id=self.task_id,
            current_node_id=self.current_node_id,
            discovery_path=self.discovery_path,
            steps=self.steps,
            max_steps=self.max_steps,
            done=self.done,
            reward=self.reward,
        )

    def _build_observation(self) -> ResearchLibrarianObservation:
        node = self.nodes.get(self.current_node_id, {})
        task = self.tasks.get(self.task_id, {})

        return ResearchLibrarianObservation(
            current_node_id=node.get("node_id"),
            current_title=node.get("title", ""),
            current_domain=node.get("domain", ""),
            current_content=node.get("content_text", ""),
            available_citations=node.get("citations", []),
            discovery_path=self.discovery_path,
            indexed_count=len(self.nodes),
            reward=self.reward,
            done=self.done,
            task_id=self.task_id,
            task_description=task.get("description", ""),
        )