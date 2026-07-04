"""Moderator = the orchestration brain. Runs on the SMALL model with strict JSON
outputs, so steering the debate costs a fraction of a persona turn."""
from typing import Dict, List

import config
from core.llm import chat_json
from .base import BaseAgent

FRAME_SYSTEM = """You are the sharpest creative strategist in India moderating a \
marketing roundtable of legendary copywriters. Your framing decides whether the \
debate produces genius or mush. Be specific to THIS brief — no generic questions."""

FRAME_USER = """BRIEF: {brief}

EXPERTS AT THE TABLE:
{roster}

Return JSON:
{{
  "reframed_brief": "the brief restated as a sharp creative problem in one sentence",
  "audience_insight": "one non-obvious truth about this target audience",
  "debate_questions": ["3 provocative questions that will force the experts to clash"],
  "formats": ["deliverable formats this brief actually needs, from: reel, linkedin, ad_copy, campaign"]
}}"""

REVIEW_SYSTEM = """You are the moderator of a creative roundtable. You are ruthless \
about quality: reward specificity and originality, attack vagueness, cliché, and \
ideas that only work in a conference room. Your challenges must create productive \
conflict — pit experts' philosophies against each other."""

REVIEW_USER = """BRIEF: {brief}

ROUND {round} TRANSCRIPT:
{transcript}

PREVIOUS SUMMARY (if any):
{summary}

Return JSON:
{{
  "summary": "the state of the debate in under 120 words — strongest ideas, live disagreements, what's still weak",
  "spotlight": "the single strongest idea on the table right now, in one line",
  "challenges": {{"<expert name>": "a direct, specific challenge for them — attack a weakness in their idea or force them to respond to a rival's"}},
  "verdict": "continue" or "converge"
}}

Rules for challenges: one per expert, each must name a concrete flaw or rival idea. \
Verdict "converge" only when ideas have stopped improving or a clear winner combination exists."""


class ModeratorAgent(BaseAgent):
    def frame(self, brief: str, personas: List[Dict]) -> Dict:
        roster = "\n".join(f"- {p['name']}" for p in personas)
        return chat_json(
            system=FRAME_SYSTEM,
            user=FRAME_USER.format(brief=brief, roster=roster),
            tier="small",
            max_tokens=config.CAP_MODERATOR,
            temperature=0.6,
        )

    def review(self, brief: str, round_num: int,
               transcript: List[Dict], summary: str) -> Dict:
        text = "\n\n".join(f"{m['role']}:\n{m['content']}" for m in transcript)
        return chat_json(
            system=REVIEW_SYSTEM,
            user=REVIEW_USER.format(brief=brief, round=round_num,
                                    transcript=text, summary=summary or "None"),
            tier="small",
            max_tokens=config.CAP_MODERATOR,
            temperature=0.4,
        )

    def invoke(self, input_data: Dict) -> Dict:
        return self.frame(input_data["brief"], input_data.get("personas", []))
