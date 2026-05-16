import json
from pathlib import Path
from typing import Optional

import streamlit as st


@st.cache_data
def load_skills_index(index_path: str) -> dict:
    path = Path(index_path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_skill(prompt: str, skills: dict) -> Optional[str]:
    """
    Lightweight keyword router.
    Fast enough for MVP and avoids calling the LLM just to select a skill.
    """
    p = (prompt or "").lower()
    best_key = None
    best_score = 0

    for key, skill in skills.items():
        triggers = skill.get("triggers", [])
        priority = skill.get("priority", 0)

        trigger_score = sum(1 for t in triggers if t.lower() in p)

        if trigger_score <= 0:
            continue

        final_score = trigger_score * 100 + priority

        if final_score > best_score:
            best_key = key
            best_score = final_score

    return best_key


@st.cache_data
def load_skill_text(skills_dir: str, skill_file: str) -> str:
    path = Path(skills_dir) / skill_file

    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")


def get_skill_instruction(prompt: str, skills_dir: str, index_path: str) -> tuple[str | None, str]:
    skills = load_skills_index(index_path)
    skill_key = select_skill(prompt, skills)

    if not skill_key:
        return None, ""

    skill_file = skills.get(skill_key, {}).get("file", "")

    if not skill_file:
        return skill_key, ""

    skill_text = load_skill_text(skills_dir, skill_file)
    return skill_key, skill_text
