import re


def normalize(text: str) -> str:
    return text.strip().lower()


def grade_research_discovery(task_id: str, agent_answer: str) -> float:
    answer = normalize(agent_answer)

    if task_id == "identify_technology":
        if "surface plasmon resonance" in answer:
            return 1.0
        if "plasmon resonance" in answer:
            return 0.7
        return 0.0

    if task_id == "chemical_ratio":
        ratio_patterns = [
            r"\b7\s*:\s*3\b",
            r"\b7\s*/\s*3\b",
            r"\b7\s*to\s*3\b",
            r"\bseven\s*to\s*three\b",
        ]
        for pattern in ratio_patterns:
            if re.search(pattern, answer):
                return 1.0
        return 0.0

    if task_id == "final_synthesis":
        has_ratio = any(
            re.search(pattern, answer)
            for pattern in [
                r"\b7\s*:\s*3\b",
                r"\b7\s*/\s*3\b",
                r"\b7\s*to\s*3\b",
                r"\bseven\s*to\s*three\b",
            ]
        )
        has_artifact = "art-402" in answer or "artifact 402" in answer

        if has_ratio and has_artifact:
            return 1.0
        if has_ratio or has_artifact:
            return 0.5
        return 0.0

    return 0.0