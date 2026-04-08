# research_library

# 🔬 Research Librarian — OpenEnv RL Environment

> Submitted to the **Meta PyTorch OpenEnv Hackathon**

A scientific discovery environment where an AI agent acts as a multi-disciplinary researcher, navigating a knowledge graph of academic papers by following citation trails across domains like Quantum Physics, Material Science, Chemistry, and Archaeology to synthesize cross-domain breakthroughs.

---

## 🌍 Environment Overview

Traditional LLM benchmarks test static recall. This environment tests **active research behavior** — can an agent follow evidence trails across disconnected domains and synthesize a conclusion it could not have reached from any single source?

The agent starts at a node in a knowledge graph and must use `search`, `read`, and `submit` actions to traverse the graph and answer a research question. It is rewarded for exploring new domains and penalized for wasted steps.

---

## 🎬 Action Space

| Action | Format | Description |
|--------|--------|-------------|
| Search | `search(query)` | Finds nodes matching a keyword query |
| Read | `read(node_id)` | Reads a specific node and follows its citations |
| Submit | `submit(answer)` | Submits a final synthesized answer for grading |

## 👁️ Observation Space

Each step returns a `ResearchLibrarianObservation` containing:

| Field | Type | Description |
|-------|------|-------------|
| `current_node_id` | str | ID of the currently active node |
| `current_title` | str | Title of the current paper/artifact |
| `current_domain` | str | Academic domain of the current node |
| `current_content` | str | Full text content of the node |
| `available_citations` | list[str] | Node IDs this paper cites |
| `discovery_path` | list[str] | Domains visited so far |
| `indexed_count` | int | Total nodes in the knowledge graph |
| `reward` | float | Cumulative reward so far |
| `done` | bool | Whether the episode has ended |
| `task_id` | str | The current task being solved |
| `task_description` | str | Natural language description of the task |

---

## 📋 Tasks

### 🟢 Easy — `identify_technology`
**Start Node:** `QUANTUMPHYS02` (Surface Plasmon Resonance paper)
**Objective:** Identify the key optical phenomenon described in the starting node.
**Expected Answer:** "Surface Plasmon Resonance"
**Max Steps:** 8
**Baseline Score:** 0.95

### 🟡 Medium — `chemical_ratio`
**Start Node:** `MATSCI03` (Material Science Node)
**Objective:** Find the critical silver-to-gold nanoparticle ratio referenced in the chemistry domain.
**Expected Answer:** Contains "73" in any form (73%, 7:3, 73 ratio)
**Max Steps:** 8
**Baseline Score:** 1.0

### 🔴 Hard — `final_synthesis`
**Start Node:** `CSMATH01` (CS / Math Node)
**Objective:** Traverse at least 4 domains and synthesize a complete answer referencing both the 73% ratio AND artifact `ART-402`.
**Expected Answer:** Must mention "73" AND "ART-402"
**Max Steps:** 8
**Baseline Score:** 1.0

---

## 🏆 Reward Function

| Event | Reward |
|-------|--------|
| Each step taken | -0.05 (time penalty) |
| Search that finds a matching node | +0.20 |
| Read that visits a new domain | +0.25 |
| Any successful tool use | +0.10 |
| Correct `submit` answer | +1.0 / +0.7 / +0.5 based on completeness |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start the environment server
uvicorn app:app --host 0.0.0.0 --port 8000

# Run the deterministic baseline agent (new terminal)
python baselineagent.py
```

### Run Inference (LLM Agent)

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py
```

### Docker

```bash
docker build -t research-librarian .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1/ \
  research-librarian
```

---

## 📊 Baseline Performance

| Task | Difficulty | Baseline Score | Steps Used |
|------|-----------|---------------|------------|
| `identify_technology` | Easy | 0.95 | 1 |
| `chemical_ratio` | Medium | 1.0 | 2 |
| `final_synthesis` | Hard | 1.0 | 5 |

---

## 📂 Project Structure
