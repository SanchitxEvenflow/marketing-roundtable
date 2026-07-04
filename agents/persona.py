from typing import Dict

import config
from core.llm import chat
from .base import BaseAgent
from .distiller import build_style_card

PERSONA_SYSTEM = """You are {name} — the real person, sitting at a live creative \
roundtable. You are NOT an AI describing {name}; you ARE them.

YOUR VOICE CARD:
{card}

RULES OF THE TABLE:
- Speak exactly as {name} would: vocabulary, rhythm, Hinglish ratio, humor style.
- Be opinionated. Take a position. Disagree loudly when your philosophy disagrees.
- Ground every idea in craft: name the hook, the emotional trigger, the platform \
mechanic that makes it spread.
- India-first instincts: culture, cricket, Bollywood, festivals, street slang — but \
only where authentic to YOUR style.
- Never be generic. If an idea could come from any marketer, kill it and go weirder.
- No preamble, no "As {name}, I think...". Just talk, like a transcript."""

PITCH_USER = """THE BRIEF: {brief}

MODERATOR'S FRAMING:
{framing}

Give your opening pitch. Maximum 200 words, in this shape:
THE BIG IDEA — one line
THE HOOK — the actual first 3 seconds / first line, written verbatim
WHY IT WORKS — the psychology + platform logic, 2-3 sentences
THE TRAP — the safe, obvious approach everyone else will pitch, and why it fails"""

CLASH_USER = """THE BRIEF: {brief}

WHERE THE DEBATE STANDS (moderator's summary):
{summary}

THE MODERATOR CHALLENGES YOU DIRECTLY:
{challenge}

React as only you would. Maximum 150 words. You may defend, steal-and-improve \
someone else's idea, or concede-and-flip. Reference other speakers by name. \
End with your sharpened idea in ONE line prefixed "FINAL:"."""


class PersonaAgent(BaseAgent):
    def __init__(self, name: str, profile: str):
        self.name = name
        self.card = build_style_card(name, profile)
        self.system = PERSONA_SYSTEM.format(name=name, card=self.card)

    def pitch(self, brief: str, framing: str) -> Dict:
        content = chat(
            system=self.system,
            user=PITCH_USER.format(brief=brief, framing=framing),
            tier="large",
            max_tokens=config.CAP_PITCH,
            temperature=0.9,
        )
        return {"role": self.name, "content": content}

    def clash(self, brief: str, summary: str, challenge: str) -> Dict:
        content = chat(
            system=self.system,
            user=CLASH_USER.format(brief=brief, summary=summary, challenge=challenge),
            tier="large",
            max_tokens=config.CAP_CLASH,
            temperature=0.9,
        )
        return {"role": self.name, "content": content}

    def invoke(self, input_data: Dict) -> Dict:
        return self.pitch(input_data["brief"], input_data.get("framing", ""))
