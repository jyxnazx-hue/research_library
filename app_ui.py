import asyncio
import gradio as gr

from models import SearchAction, ReadAction, SubmitAction
from server.research_librarian_environment import ResearchLibrarianEnvironment

env = ResearchLibrarianEnvironment()


async def run_action(task_id: str, action_type: str, input_text: str):
    global env

    if env.task_id != task_id or env.current_node_id is None:
        obs = await env.reset(task_id)
    else:
        obs = env._build_observation()

    if action_type == "Search":
        obs = await env.step(SearchAction(query=input_text))
    elif action_type == "Read":
        obs = await env.step(ReadAction(node_id=input_text))
    elif action_type == "Submit":
        obs = await env.step(SubmitAction(answer=input_text))

    discovery_md = "### Discovery Path\n" + " → ".join(obs.discovery_path)
    stats_md = (
        f"**Stats:** {obs.indexed_count} nodes indexed | "
        f"Reward: {obs.reward:.2f} | Done: {obs.done}"
    )

    node_md = f"""
### Current Node
**ID:** {obs.current_node_id}

**Title:** {obs.current_title}

**Domain:** {obs.current_domain}

**Content:**  
{obs.current_content}

**Citations:** {", ".join(obs.available_citations) if obs.available_citations else "None"}
"""

    return discovery_md, stats_md, node_md


def sync_run(task_id: str, action_type: str, input_text: str):
    return asyncio.run(run_action(task_id, action_type, input_text))


with gr.Blocks(title="Scientific Discovery Research Librarian") as demo:
    gr.Markdown("# Scientific Discovery Research Librarian")
    gr.Markdown(
        "Traverse physics, materials science, chemistry, and archaeology "
        "to solve the optical nano-filter discovery challenge."
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_id = gr.Dropdown(
                choices=[
                    "identify_technology",
                    "chemical_ratio",
                    "final_synthesis",
                ],
                value="identify_technology",
                label="Task",
            )

            action_type = gr.Radio(
                ["Search", "Read", "Submit"],
                label="Action Type",
                value="Search",
            )

            input_text = gr.Textbox(
                label="Input",
                placeholder="Enter query, node ID, or final answer..."
            )

            run_btn = gr.Button("Execute", variant="primary")

        with gr.Column(scale=2):
            discovery_tracker = gr.Markdown("### Discovery Path\nStart")
            system_stats = gr.Markdown("**Stats:** 0 nodes indexed | Reward: 0.00")
            node_view = gr.Markdown("### Current Node\nNo node loaded yet.")

    run_btn.click(
        fn=sync_run,
        inputs=[task_id, action_type, input_text],
        outputs=[discovery_tracker, system_stats, node_view],
    )


if __name__ == "__main__":
    demo.launch()