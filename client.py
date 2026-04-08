from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    ActionType,
    ResearchLibrarianObservation,
    ResearchLibrarianState,
)


class ResearchLibrarianEnv(
    EnvClient[ActionType, ResearchLibrarianObservation, ResearchLibrarianState]
):
    def _step_payload(self, action: ActionType) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[ResearchLibrarianObservation]:
        obs_data = payload.get("observation", {})
        observation = ResearchLibrarianObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ResearchLibrarianState:
        return ResearchLibrarianState(**payload)