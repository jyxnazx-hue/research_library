import asyncio

from models import ReadAction, SubmitAction
from server.research_librarian_environment import ResearchLibrarianEnvironment


async def run_easy():
    env = ResearchLibrarianEnvironment()
    obs = await env.reset("identify_technology")
    print("\n--- EASY TASK ---")
    print("Start:", obs.current_node_id, "|", obs.current_title)

    obs = await env.step(
        SubmitAction(answer="Surface Plasmon Resonance")
    )
    print("Reward:", obs.reward)
    print("Done:", obs.done)


async def run_medium():
    env = ResearchLibrarianEnvironment()
    obs = await env.reset("chemical_ratio")
    print("\n--- MEDIUM TASK ---")
    print("Start:", obs.current_node_id, "|", obs.current_title)

    obs = await env.step(ReadAction(node_id="CHEM73RATIO"))
    print("Visited:", obs.current_node_id, "|", obs.current_title)

    obs = await env.step(
        SubmitAction(answer="7:3 silver to gold")
    )
    print("Reward:", obs.reward)
    print("Done:", obs.done)


async def run_hard():
    env = ResearchLibrarianEnvironment()
    obs = await env.reset("final_synthesis")
    print("\n--- HARD TASK ---")
    print("Start:", obs.current_node_id, "|", obs.current_title)

    path = [
        "QUANTUMPHYS02",
        "MATSCI03",
        "CHEM73RATIO",
        "ARCHART402",
    ]

    for node_id in path:
        obs = await env.step(ReadAction(node_id=node_id))
        print("Visited:", obs.current_node_id, "|", obs.current_domain)

    obs = await env.step(
        SubmitAction(
            answer="ART-402 supports a 7:3 silver-gold ratio for the optical nano-filter through surface plasmon resonance."
        )
    )
    print("Reward:", obs.reward)
    print("Done:", obs.done)
    print("Path:", obs.discovery_path)


async def main():
    await run_easy()
    await run_medium()
    await run_hard()


if __name__ == "__main__":
    asyncio.run(main())