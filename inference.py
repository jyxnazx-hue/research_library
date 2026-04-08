import os
import json
import asyncio
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from models import SearchAction, ReadAction, SubmitAction
from server.research_librarian_environment import ResearchLibrarianEnvironment


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

ENV_NAME = "research_librarian"


SYSTEM_PROMPT = """
You are an agent solving a research task.

Return ONLY one JSON object in exactly one of these forms:
{"action_type":"search","query":"..."}
{"action_type":"read","node_id":"..."}
{"action_type":"submit","answer":"..."}

Rules:
- Output JSON only.
- No markdown.
- No explanation.
- Choose exactly one action.
- Use the task description and current content.
- If you already know the answer, submit.
- If a specific node is useful, read it.
- Otherwise search.
""".strip()


def build_user_prompt(obs):
    return f"""
Task ID: {obs.task_id}
Task Description: {obs.task_description}

Current Node ID: {obs.current_node_id}
Current Title: {obs.current_title}
Current Domain: {obs.current_domain}
Current Content:
{obs.current_content}

Available Citations: {obs.available_citations}
Discovery Path: {obs.discovery_path}
Indexed Count: {obs.indexed_count}
Reward: {obs.reward}
Done: {obs.done}

Choose the single best next action.
Return JSON only.
""".strip()


def extract_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model output: {text}")

    return text[start:end + 1]


def parse_action(raw_text: str):
    data = json.loads(extract_json(raw_text))
    action_type = data.get("action_type")

    if action_type == "search":
        return SearchAction(action_type="search", query=data["query"])
    elif action_type == "read":
        return ReadAction(action_type="read", node_id=data["node_id"])
    elif action_type == "submit":
        return SubmitAction(action_type="submit", answer=data["answer"])
    else:
        raise ValueError(f"Unknown action_type: {action_type}")


def format_action(action) -> str:
    if isinstance(action, SearchAction):
        return f'search("{action.query}")'
    if isinstance(action, ReadAction):
        return f'read("{action.node_id}")'
    if isinstance(action, SubmitAction):
        answer = action.answer.replace("\n", " ").replace('"', "'")
        return f'submit("{answer[:80]}")'
    return "unknown()"


def fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def fmt_reward(value: float) -> str:
    return f"{value:.2f}"


async def run_task(task_name: str):
    env = ResearchLibrarianEnvironment()
    rewards = []
    steps_taken = 0
    success = False

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

    try:
        obs = await env.reset(task_name)

        max_steps = 8
        for step_num in range(1, max_steps + 1):
            if obs.done:
                break

            error_msg = "null"
            action = None

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(obs)},
                    ],
                    temperature=0,
                )

                raw_text = response.choices[0].message.content
                action = parse_action(raw_text)

            except Exception as e:
                error_msg = str(e).replace("\n", " ").strip()
                action = SubmitAction(
                    action_type="submit",
                    answer=f"Failed due to inference error: {error_msg}"
                )

            try:
                obs = await env.step(action)
            except Exception as e:
                error_msg = str(e).replace("\n", " ").strip()
                rewards.append(0.00)
                steps_taken = step_num
                print(
                    f"[STEP] step={step_num} action={format_action(action)} "
                    f"reward=0.00 done=true error={error_msg}"
                )
                break

            step_reward = obs.reward - sum(rewards)
            rewards.append(step_reward)
            steps_taken = step_num

            print(
                f"[STEP] step={step_num} action={format_action(action)} "
                f"reward={fmt_reward(step_reward)} done={fmt_bool(obs.done)} error={error_msg}"
            )

            if obs.done:
                success = obs.reward > 0
                break

    except Exception:
        success = False

    rewards_str = ",".join(fmt_reward(r) for r in rewards)
    print(f"[END] success={fmt_bool(success)} steps={steps_taken} rewards={rewards_str}")


async def main():
    task_names = [
        "identify_technology",
        "chemical_ratio",
        "final_synthesis",
    ]

    for task_name in task_names:
        await run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())