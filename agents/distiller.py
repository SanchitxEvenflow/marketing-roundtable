"""Distill a 10-20KB research profile into a ~300-word "voice card".

This is the single biggest token saver in the system: the full profile passes
through the LLM exactly once per persona (ever — the card is cached on disk
keyed by profile hash). Every debate turn then costs ~350 tokens of identity
instead of ~4,000.
"""
import hashlib
import json
import os

import config
from core.llm import chat

DISTILL_SYSTEM = """You are building a compact VOICE CARD so an AI agent can \
convincingly speak as a real Indian marketing/copywriting legend in a live creative debate.
Extract ONLY what is needed to reproduce their voice and thinking. Be specific and \
concrete — named techniques, real quirks, actual beliefs — never generic filler like \
"engaging" or "creative"."""

DISTILL_USER = """RESEARCH PROFILE OF {name}:
{profile}

Produce the voice card in exactly this skeleton, total under 300 words:

IDENTITY: one line — who they are, what made them famous
VOICE: tone, rhythm, Hinglish ratio, sentence length, humor type
SIGNATURE MOVES: 4-6 named techniques, each with a tiny concrete example
BELIEFS: 3-4 hills they die on about marketing/creativity
HOOK PATTERNS: the 3 opening patterns they reach for first
NEVER: 2-3 things they would never do or say
SAMPLE LINE: one invented line that sounds exactly like them"""


def build_style_card(name: str, profile: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    key = hashlib.md5(profile.encode()).hexdigest()[:16]
    cache_path = os.path.join(config.CACHE_DIR, f"{key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)["card"]

    card = chat(
        system=DISTILL_SYSTEM,
        user=DISTILL_USER.format(name=name, profile=profile[:16000]),
        tier="large",
        max_tokens=config.CAP_STYLE_CARD,
        temperature=0.4,
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "card": card}, f, ensure_ascii=False, indent=2)
    return card
