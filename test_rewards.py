import asyncio

from models import ReadAction, SubmitAction
from server.research_librarian_environment import ResearchLibrarianEnvironment

def test_identify_technology_correct():
    async def run():
        env = ResearchLibrarianEnvironment()
        obs = await env.reset("identify_technology")
        assert obs.current_node_id == "QUANTUM_PHYS_02"
        assert obs.reward == 0.0
        assert obs.done is False

        obs = await env.step(
            SubmitAction(answer="Surface Plasmon Resonance")
        )
        assert obs.done is True
        assert obs.reward >= 0.95

    asyncio.run(run())

def test_identify_technology_wrong():
    async def run():
        env = ResearchLibrarianEnvironment()
        obs = await env.reset("identify_technology")

        obs = await env.step(
            SubmitAction(answer="Copper Wires")
        )
        assert obs.done is True
        assert obs.reward == 0.0

    asyncio.run(run())

def test_read_increases_reward():
    async def run():
        env = ResearchLibrarianEnvironment()
        obs = await env.reset("identify_technology")

        obs = await env.step(
            ReadAction(node_id="CS_MATH_01")
        )
        assert obs.current_node_id == "CS_MATH_01"
        assert obs.reward > 0.0
        assert obs.done is False

    asyncio.run(run())

def test_chemical_ratio_correct():
    async def run():
        env = ResearchLibrarianEnvironment()
        obs = await env.reset("chemical_ratio")

        obs = await env.step(
            SubmitAction(answer="7:3 silver to gold")
        )
        assert obs.done is True
        assert obs.reward >= 0.95

    asyncio.run(run())

def test_final_synthesis_partial():
    async def run():
        env = ResearchLibrarianEnvironment()
        obs = await env.reset("final_synthesis")

        obs = await env.step(
            SubmitAction(answer="ART-402 seems important")
        )
        assert obs.done is True
        assert 0.4 <= obs.reward <= 0.5

    asyncio.run(run())