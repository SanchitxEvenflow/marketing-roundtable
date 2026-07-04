from typing import Dict, List

import config
from core.llm import chat
from .base import BaseAgent

SYNTH_SYSTEM = """You are the Creative Director closing a roundtable of India's best \
marketing minds. Your job is synthesis, not summary: steal the strongest thinking \
from every expert, fuse conflicting ideas into something better than any single one, \
and make hard choices. A brief that includes everything is a brief that says nothing. \
Preserve the boldness — do not sand the ideas down into agency-safe mush."""

SYNTH_USER = """BRIEF: {brief}

MODERATOR'S FRAMING:
{framing}

FULL DEBATE TRANSCRIPT:
{transcript}

Write the definitive CREATIVE STRATEGY BRIEF in markdown:

## The Winning Territory
The single creative direction we're committing to, and which experts' thinking it fuses. 3-4 sentences.

## The Big Idea
One line. Then 2-3 sentences on the psychology of why it spreads.

## Message Architecture
- Core message (one line)
- 3 supporting angles, each tagged with the emotion it triggers

## Hook Bank
10 hooks, written verbatim, ready to use. Mix formats: question, confession, absurd claim, pattern-interrupt, cultural reference. Credit the expert whose style each hook channels.

## Rejected Directions
2-3 ideas from the debate we're deliberately NOT pursuing, and the one-line reason. \
(This protects the strategy from dilution later.)

## Format Notes
For each deliverable format, one line on how the big idea should bend for that format."""


class CreativeDirectorAgent(BaseAgent):
    def synthesize(self, brief: str, framing: str, transcript: List[Dict]) -> str:
        text = "\n\n".join(
            f"{m['role']}:\n{m['content'][:1500]}" for m in transcript
        )
        return chat(
            system=SYNTH_SYSTEM,
            user=SYNTH_USER.format(brief=brief, framing=framing, transcript=text),
            tier="large",
            max_tokens=config.CAP_SYNTHESIS,
            temperature=0.7,
        )

    def invoke(self, input_data: Dict) -> Dict:
        content = self.synthesize(
            input_data["brief"],
            input_data.get("framing", ""),
            input_data.get("messages", []),
        )
        return {"role": "Creative Director", "content": content}
