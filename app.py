# server/app.py

from fastapi import FastAPI, Body # Add Body to imports
from pydantic import BaseModel
from typing import Optional

from models import ActionType
from server.research_librarian_environment import ResearchLibraryEnvironment

app = FastAPI(title="Research Librarian API")
env = ResearchLibraryEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- UPDATED THIS ENDPOINT ---
@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    # Use the task_id if provided, otherwise default to None
    task_id = req.task_id if req else None
    obs = await env.reset(task_id)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }

@app.post("/step")
async def step(action: ActionType):
    obs = await env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }
