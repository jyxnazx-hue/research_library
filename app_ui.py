import asyncio
import gradio as gr
from models import SearchAction, ReadAction, SubmitAction
from server.environment import ResearchLibrarianEnvironment

env = ResearchLibrarianEnvironment()

# DESIGN: Blue/Green Bubble Sanctuary
css = """
body { 
    background: radial-gradient(circle at center, #1e3a8a, #064e3b); 
    overflow: hidden; 
}
/* Floating Bubble Animation */
.bubbles { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }
.bubble { position: absolute; border-radius: 50%; background: rgba(255, 255, 255, 0.1); animation: float 15s infinite; }
@keyframes float { 0% { transform: translateY(100vh) scale(0); } 100% { transform: translateY(-10vh) scale(1.5); } }

/* Main White Interface Box */
.main-interface {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    color: #1f2937;
    z-index: 10;
    margin-top: 5vh;
}
.holographic-btn { background: linear-gradient(90deg, #10b981, #3b82f6) !important; color: white !important; }
"""

async def run_bci_lab(task_id, mode, val):
    if env.current_node_id is None or env.task_id != task_id:
        obs = await env.reset(task_id)
    
    if mode == "Search": action = SearchAction(query=val)
    elif mode == "Read": action = ReadAction(node_id=val)
    else: action = SubmitAction(answer=val)
    
    obs = await env.step(action)
    
    # CANVAS BOARD Presentation
    canvas_html = f"""
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;'>
        <div style='background: #f3f4f6; padding: 1rem; border-radius: 10px;'>
            <h4>📑 Synthesis Found</h4>
            <p>{obs.current_content}</p>
        </div>
        <div style='background: #ecfdf5; padding: 1rem; border-radius: 10px; border-left: 5px solid #10b981;'>
            <h4>🔬 Lab Stats</h4>
            <b>Domains Searched:</b> {len(obs.discovery_path)}/10<br>
            <b>Citations Indexed:</b> {len(obs.available_citations) * 12} nodes<br>
            <b>Current Domain:</b> {obs.current_domain}
        </div>
    </div>
    """
    
    # Order of findings
    findings_list = "\n".join([f"{i+1}. [{d}] node accessed" for i, d in enumerate(obs.discovery_path)])
    
    return obs.graph_mermaid, canvas_html, findings_list, f"**REWARD:** {obs.reward:.2f}"

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    # Background Bubbles
    gr.HTML("<div class='bubbles'>" + "".join([f"<div class='bubble' style='left:{i*10}%; width:{i*5}px; height:{i*5}px; animation-delay:{i}s'></div>" for i in range(10)]) + "</div>")
    
    with gr.Column(elem_classes="main-interface"):
        gr.HTML("<h1 style='text-align: center; color: #1e3a8a;'>🔬 BCI DISCOVERY SANCTUARY</h1>")
        
        with gr.Row():
            # CONTROL PANEL
            with gr.Column(scale=1):
                task = gr.Dropdown(["neural_bypass", "silent_speech", "prosthetic_sync"], value="neural_bypass", label="Mission")
                mode = gr.Radio(["Search", "Read", "Submit"], label="Neural Mode", value="Search")
                inp = gr.Textbox(label="Command Input", placeholder="Keywords (e.g. 'Stentrode', 'EEG noise')...")
                run_btn = gr.Button("⚡ Execute Discovery", elem_classes="holographic-btn")
                reward_out = gr.Markdown("**REWARD:** 0.01")
            
            # THE CANVAS BOARD
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Discovery Canvas"):
                        canvas_out = gr.HTML("<p>Awaiting Lab Initialization...</p>")
                    with gr.TabItem("🕸️ Knowledge Graph"):
                        graph_out = gr.Markdown("```mermaid\ngraph LR\nStart\n```")
                    with gr.TabItem("📜 Step History"):
                        history_out = gr.Markdown("No steps taken yet.")

    run_btn.click(lambda t, m, v: asyncio.run(run_bci_lab(t, m, v)), [task, mode, inp], [graph_out, canvas_out, history_out, reward_out])

if __name__ == "__main__":
    demo.launch()
