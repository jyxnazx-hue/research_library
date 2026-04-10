import asyncio
import gradio as gr
from models import SearchAction, ReadAction, SubmitAction
from server.research_librarian_environment import ResearchLibrarianEnvironment

env = ResearchLibrarianEnvironment()

async def run_discovery(task_id, mode, val):
    if env.current_node_id is None or env.task_id != task_id:
        obs = await env.reset(task_id)
    
    if mode == "Search": action = SearchAction(query=val)
    elif mode == "Read": action = ReadAction(node_id=val)
    else: action = SubmitAction(answer=val)
    
    obs = await env.step(action)
    
    # Format the Synthesis Report
    report_md = f"""### 📑 {obs.current_title if obs.current_title else 'Awaiting Telemetry...'}
**Domain:** {obs.current_domain} | **Reward:** {obs.reward:.2f}

#### 🔬 Content Summary
{obs.current_content if obs.current_content else 'No research node active.'}

#### 🔗 Citations Found
{chr(10).join([f"* {c}" for c in obs.available_citations]) if obs.available_citations else "*None detected.*"}
"""
    log_md = "\n".join([f"> {t}" for t in obs.thought_log])
    
    return obs.graph_mermaid, report_md, log_md, f"**Reward:** {obs.reward:.2f}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:
    gr.Markdown("# 🔬 Scientific Discovery Research Librarian")
    gr.HTML("<hr>")

    with gr.Row():
        # LEFT: CONTROLS
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 🛠️ Lab Mission")
            task = gr.Radio(["identify_technology", "chemical_ratio", "final_synthesis"], value="identify_technology", label="Mission Select")
            gr.HTML("<hr>")
            mode = gr.Radio(["Search", "Read", "Submit"], label="Action Mode", value="Search")
            inp = gr.Textbox(label="Neural Command", placeholder="Keywords or Node IDs...")
            run_btn = gr.Button("⚡ Execute Discovery", variant="primary")
            
            gr.Markdown("### 📡 Neural Trace")
            log_out = gr.Markdown("> Awaiting Neural Link...")
            reward_out = gr.Markdown("**Reward:** 0.01")

        # RIGHT: SYNTHESIS & GRAPH
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📄 Synthesis Engine"):
                    report_out = gr.Markdown("### 📡 System Ready\nSelect a mission to begin.")
                with gr.TabItem("🕸️ Citation Graph"):
                    graph_out = gr.Markdown("```mermaid\ngraph LR\nStart\n```")

    run_btn.click(
        fn=lambda t, m, v: asyncio.run(run_discovery(t, m, v)),
        inputs=[task, mode, inp],
        outputs=[graph_out, report_out, log_out, reward_out]
    )

if __name__ == "__main__":
    demo.launch()
