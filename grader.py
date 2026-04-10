def grade_research_discovery(task_id: str, answer: str) -> float:
    # Baseline 0.05 to avoid the "0.0 is out of range" error
    score = 0.05 
    
    answer_norm = answer.lower()
    
    if task_id == "chemical_ratio":
        if "7:3" in answer or "70:30" in answer:
            score += 0.80
        if "silver" in answer_norm and "gold" in answer_norm:
            score += 0.10
            
    elif task_id == "identify_technology":
        if "surface plasmon" in answer_norm or "resonance" in answer_norm:
            score += 0.85

    # Clamp logic: never 0.0, never 1.0. Strictly (0, 1).
    return max(0.01, min(0.99, score))
