import json
from pathlib import Path
from typing import Dict, Optional
from models import ActionType, LibraryObservation, ReadAction, ResearchNode, SearchAction, SubmitAction
from server.grader import grade_research_discovery

class ResearchLibrarianEnvironment:
    def __init__(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        with open(data_dir / "nodes.json", "r", encoding="utf-8") as f:
            nodes = json.load(f)
        with open(data_dir / "tasks.json", "r", encoding="utf-8") as f:
            tasks = json.load(f)

        self.nodes = {n["node_id"]: ResearchNode(**n) for n in nodes}
        self.tasks = {t["task_id"]: t for t in tasks}
        self.reset_state()

    def reset_state(self):
        self.task_id = "identify_technology"
        self.current_node_id = None
        self.discovery_path = []
        self.thought_log = ["📡 Neural Link Established. Ready for discovery."]
        self.reward = 0.01
        self.done = False

    async def reset(self, task_id: Optional[str] = None) -> LibraryObservation:
        self.reset_state()
        self.task_id = task_id or "identify_technology"
        task = self.tasks[self.task_id]
        self.current_node_id = task["start_node_id"]
        
        start_node = self.nodes[self.current_node_id]
        self.discovery_path.append(start_node.domain)
        self.thought_log.append(f"🧠 Initialized mission in {start_node.domain} domain.")
        
        return self._build_observation("Environment Reset Complete.")

    async def step(self, action: ActionType) -> LibraryObservation:
        msg = f"Executing {action.action_type}..."
        
        if isinstance(action, SearchAction):
            query = action.query.lower()
            self.thought_log.append(f"🔍 Scanning library for: '{query}'")
            # Basic keyword search logic
            for node in self.nodes.values():
                if query in node.content_text.lower() or query in node.title.lower():
                    self.current_node_id = node.node_id
                    if node.domain not in self.discovery_path:
                        self.discovery_path.append(node.domain)
                        self.thought_log.append(f"🚀 INTERDISCIPLINARY JUMP: {node.domain} reached.")
                    break
                    
        elif isinstance(action, SubmitAction):
            self.reward = grade_research_discovery(self.task_id, action.answer)
            self.done = True
            self.thought_log.append(f"🏁 Synthesis Submitted. Final Score: {self.reward}")

        return self._build_observation(msg)

    def _build_observation(self, message: str) -> LibraryObservation:
        curr = self.nodes.get(self.current_node_id)
        
        # Build Mermaid Graph String
        graph = "graph LR\nStart"
        if self.discovery_path:
            for i, domain in enumerate(self.discovery_path):
                graph += f" --> D{i}[{domain}]"
        
        return LibraryObservation(
            current_node_id=self.current_node_id,
            current_title=curr.title if curr else "",
            current_content=curr.content_text if curr else "",
            current_domain=curr.domain if curr else "",
            available_citations=curr.citations if curr else [],
            discovery_path=self.discovery_path,
            thought_log=self.thought_log[-5:], # Show last 5 thoughts
            graph_mermaid=f"```mermaid\n{graph}\n```",
            reward=self.reward,
            done=self.done,
            task_id=self.task_id,
            task_description=self.tasks[self.task_id]["description"]
        )
