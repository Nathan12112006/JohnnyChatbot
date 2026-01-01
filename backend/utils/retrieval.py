import json
from pathlib import Path

# Get the absolute path to the backend/data directory
DATA_DIR = Path(__file__).parent.parent / "data"

def load_json(filename):
    with open(DATA_DIR / filename) as f:
        return json.load(f)

RESUME = load_json("resume.json")
PROJECTS = load_json("projects.json")

def retrieve_context(intent: str) -> str:
    if intent == "projects":
        return json.dumps(PROJECTS)
    if intent == "skills":
        return json.dumps(RESUME.get("skills", []))
    if intent == "courses":
        return json.dumps(RESUME.get("courses", []))

    return json.dumps(RESUME)
