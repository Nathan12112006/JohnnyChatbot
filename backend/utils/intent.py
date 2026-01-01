def classify_intent(message: str) -> str:
    msg = message.lower()

    if "project" in msg:
        return "projects"
    if "skill" in msg or "technology" in msg:
        return "skills"
    if "hire" in msg or "intern" in msg or "fit" in msg:
        return "fit"
    if "class" in msg or "course" in msg:
        return "courses"
    if "approach" in msg or "solve" in msg:
        return "problem_solving"

    return "general"
