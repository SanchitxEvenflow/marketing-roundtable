"""Fallback persona source: load profiles straight from the research markdown
and let the small LLM pick who belongs at the table for this brief."""
import os
import re

import config
from core.llm import chat_json


def _extract_name(content: str, filename: str) -> str:
    match = re.search(r"^#\s*(.+?)\s*Copywrit\w*\s*Profile", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace("-copywriter-profile", "").replace(".md", "").replace("-", " ").title()


def load_all_profiles() -> list[dict]:
    profiles = []
    folder = config.PERSONA_DIR
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            content = f.read()
        profiles.append({"name": _extract_name(content, filename), "profile": content})
    return profiles


def get_relevant_personas(topic: str, k: int = 5) -> list[dict]:
    profiles = load_all_profiles()
    if len(profiles) <= k:
        return profiles

    roster = "\n".join(f"- {p['name']}: {p['profile'][:200]}" for p in profiles)
    decision = chat_json(
        system=("You are casting a creative-marketing roundtable. Pick the experts whose "
                "styles best fit the brief AND complement each other (mix humor, emotion, "
                "strategy, virality — not four of the same flavor)."),
        user=f"BRIEF: {topic}\n\nAVAILABLE EXPERTS:\n{roster}\n\n"
             f'Pick exactly {k}. Return {{"selected": ["Name", ...]}}',
        max_tokens=200,
    )
    selected = {name.lower() for name in decision.get("selected", [])}
    picked = [p for p in profiles if p["name"].lower() in selected]
    return picked[:k] if picked else profiles[:k]
