"""Format specialists turn the creative strategy brief into ready-to-ship assets.
One craft-heavy prompt per format — the format prompt carries the platform
expertise so the strategy brief doesn't have to."""
from typing import Dict

import config
from core.llm import chat
from .base import BaseAgent

FORMAT_SPECS = {
    "reel": {
        "title": "Instagram Reel",
        "system": """You are India's best short-form video scriptwriter. Your reels \
stop thumbs. You know: the first 1.5 seconds decide everything; text overlay carries \
viewers watching on mute; loops and pattern-interrupts drive rewatches; the comment \
bait is designed, not lucky.""",
        "user": """CREATIVE STRATEGY BRIEF:
{brief}

Write 2 REEL CONCEPTS. For each:

### Reel: <punchy working title>
**Duration:** 30-45s
**Hook (0-3s):** exact visual + exact spoken/overlay line
**Script:** timestamped beats (visual | dialogue/VO | text overlay) — write actual \
lines, not descriptions like "funny dialogue here"
**Audio:** trending-audio style suggestion or original sound direction
**Loop/rewatch trigger:** what makes them watch twice
**Caption:** first line must re-hook; then 1-2 lines + CTA
**Hashtags:** 5, mixed reach sizes
**Comment bait:** the deliberate detail people will argue/tag friends about""",
    },
    "linkedin": {
        "title": "LinkedIn Posts",
        "system": """You are a LinkedIn ghostwriter whose posts routinely clear 1M \
impressions in the Indian professional feed. You know: the first two lines are the \
only thing that exists before "...see more"; whitespace is rhythm; specificity beats \
polish; a post is a story or an argument, never an announcement; the ending must \
invite a take, not beg for engagement.""",
        "user": """CREATIVE STRATEGY BRIEF:
{brief}

Write 2 LINKEDIN POSTS, fully written out and ready to paste:

### Version A — Founder-voice story
Personal, specific, a real-feeling moment that lands on the brand insight. \
Hook line must create an open loop.

### Version B — Contrarian argument
Open by attacking a belief the target audience holds. Build the case in short \
punchy lines. Land on the brand's POV.

For each: the full post (with line breaks as they'd appear), then one line on \
why the hook works.""",
    },
    "ad_copy": {
        "title": "Ad Copy & Hooks",
        "system": """You are a direct-response copywriter with brand-craft instincts \
— Ogilvy discipline, Indian street fluency. Every line earns its place: concrete \
beats clever, one idea per line, the CTA continues the thought instead of breaking it.""",
        "user": """CREATIVE STRATEGY BRIEF:
{brief}

Deliver:

### Headlines (5)
Each under 8 words, each a different psychological angle (fear-of-missing, status, \
curiosity, social proof, absurd confidence). Tag the angle.

### Primary Ad Copy (3 variants)
Headline + 2-3 line body + CTA. Variant 1: emotional. Variant 2: witty/cultural. \
Variant 3: blunt value.

### Hooks Bank (8)
One-liners for video ads / statics, written verbatim, mixing Hinglish and English \
where it hits harder.""",
    },
    "campaign": {
        "title": "Campaign Plan",
        "system": """You are a campaign architect who turns one big idea into a \
multi-week, multi-platform arc. You think in phases (tease → launch → amplify → \
sustain), earned-media triggers, and creator collaboration mechanics.""",
        "user": """CREATIVE STRATEGY BRIEF:
{brief}

Design a 4-WEEK CAMPAIGN ARC:
For each phase: name, goal, key asset(s), platform focus, and the specific moment \
designed to get shared/screenshotted. End with 3 creator-collab mechanics and \
2 low-cost earned-media stunts.""",
    },
}


class FormatSpecialistAgent(BaseAgent):
    def produce(self, creative_brief: str, output_type: str) -> Dict:
        spec = FORMAT_SPECS.get(output_type)
        if spec is None:
            raise ValueError(
                f"Unknown format '{output_type}'. Available: {list(FORMAT_SPECS)}"
            )
        content = chat(
            system=spec["system"],
            user=spec["user"].format(brief=creative_brief),
            tier="large",
            max_tokens=config.CAP_FORMAT,
            temperature=0.85,
        )
        return {"role": f"Format Specialist ({spec['title']})", "content": content}

    def invoke(self, input_data: Dict) -> Dict:
        return self.produce(input_data["brief"], input_data.get("output_type", "linkedin"))
