"""LangGraph orchestration for the marketing roundtable.

Pipeline:
    cast ──► frame ──► pitch ──► clash ──► (continue? clash again : synthesize)
                                              └──► synthesize ──► produce ──► END

Token-economy design:
  - Personas carry a cached ~300-word voice card, never the full 10-20KB profile.
  - Persona turns see the moderator's rolling SUMMARY, not the full transcript.
  - The moderator runs on the small model with strict JSON output.
  - Only the one-shot Creative Director synthesis reads the whole transcript.
"""
from typing import List, TypedDict

from langgraph.graph import END, StateGraph

import config
from agents import (CreativeDirectorAgent, FormatSpecialistAgent,
                    ModeratorAgent, PersonaAgent, FORMAT_SPECS)
from retrieval import get_personas


class DebateState(TypedDict, total=False):
    brief: str
    formats: List[str]          # requested formats; [] = let the moderator decide
    num_personas: int
    max_rounds: int
    personas: List[dict]        # [{name, card}]
    framing: dict               # moderator's frame: questions, insight, formats
    transcript: List[dict]      # [{role, content}] — full log, used only at synthesis
    summary: str                # rolling debate summary (the token-cheap context)
    challenges: dict            # persona name -> current challenge
    round: int
    verdict: str                # "continue" | "converge"
    creative_brief: str
    deliverables: List[dict]


_moderator = ModeratorAgent()
_director = CreativeDirectorAgent()
_specialist = FormatSpecialistAgent()
_agents: dict[str, PersonaAgent] = {}   # built per run in cast_node


def _say(role: str, content: str):
    print(f"\n{'─' * 60}\n🎙  {role}\n{'─' * 60}\n{content}\n")


def _framing_text(framing: dict) -> str:
    return (
        f"Creative problem: {framing.get('reframed_brief', '')}\n"
        f"Audience insight: {framing.get('audience_insight', '')}\n"
        "Debate questions:\n"
        + "\n".join(f"- {q}" for q in framing.get("debate_questions", []))
    )


# ── Nodes ──────────────────────────────────────────────────────────────────

def cast_node(state: DebateState):
    """Retrieve relevant personas (Neo4j vector search) and build their agents
    with cached voice cards."""
    k = state.get("num_personas") or config.NUM_PERSONAS
    raw = get_personas(state["brief"], k=k)

    _agents.clear()
    personas = []
    for p in raw:
        print(f"[cast] preparing voice card: {p['name']}")
        agent = PersonaAgent(p["name"], p["profile"])
        _agents[p["name"]] = agent
        personas.append({"name": p["name"], "card": agent.card})
    return {"personas": personas, "transcript": [], "round": 0, "summary": ""}


def frame_node(state: DebateState):
    """Moderator reframes the brief and sets the debate agenda."""
    framing = _moderator.frame(state["brief"], state["personas"])
    formats = state.get("formats") or framing.get("formats") or config.DEFAULT_FORMATS
    formats = [f for f in formats if f in FORMAT_SPECS] or config.DEFAULT_FORMATS
    _say("Moderator (framing)", _framing_text(framing))
    return {"framing": framing, "formats": formats}


def pitch_node(state: DebateState):
    """Round 1: every expert pitches their opening take."""
    framing = _framing_text(state["framing"])
    new_messages = []
    for persona in state["personas"]:
        msg = _agents[persona["name"]].pitch(state["brief"], framing)
        _say(msg["role"], msg["content"])
        new_messages.append(msg)

    review = _moderator.review(state["brief"], 1, new_messages, "")
    _say("Moderator", f"{review['summary']}\n\n⭐ Spotlight: {review.get('spotlight', '')}")
    return {
        "transcript": state["transcript"] + new_messages
        + [{"role": "Moderator", "content": review["summary"]}],
        "summary": review["summary"],
        "challenges": review.get("challenges", {}),
        "verdict": review.get("verdict", "continue"),
        "round": 1,
    }


def clash_node(state: DebateState):
    """Clash round: each expert answers the moderator's targeted challenge.
    They see only the rolling summary + their challenge — not the transcript."""
    round_num = state["round"] + 1
    new_messages = []
    for persona in state["personas"]:
        name = persona["name"]
        challenge = state["challenges"].get(
            name, "Sharpen your idea — what makes it undeniably yours?"
        )
        msg = _agents[name].clash(state["brief"], state["summary"], challenge)
        _say(f"{msg['role']} (round {round_num})", msg["content"])
        new_messages.append(msg)

    review = _moderator.review(state["brief"], round_num, new_messages, state["summary"])
    _say("Moderator", f"{review['summary']}\n\n⭐ Spotlight: {review.get('spotlight', '')}")
    return {
        "transcript": state["transcript"] + new_messages
        + [{"role": "Moderator", "content": review["summary"]}],
        "summary": review["summary"],
        "challenges": review.get("challenges", {}),
        "verdict": review.get("verdict", "converge"),
        "round": round_num,
    }


def should_continue(state: DebateState) -> str:
    max_rounds = state.get("max_rounds") or config.MAX_CLASH_ROUNDS
    clash_rounds_done = state["round"] - 1  # round 1 was the pitch
    if state.get("verdict") == "continue" and clash_rounds_done < max_rounds:
        return "clash"
    return "synthesize"


def synthesize_node(state: DebateState):
    """Creative Director fuses the debate into the strategy brief."""
    creative_brief = _director.synthesize(
        state["brief"], _framing_text(state["framing"]), state["transcript"]
    )
    _say("Creative Director", creative_brief)
    return {"creative_brief": creative_brief}


def produce_node(state: DebateState):
    """Format specialists turn the strategy into ready-to-ship assets."""
    deliverables = []
    for fmt in state["formats"]:
        msg = _specialist.produce(state["creative_brief"], fmt)
        _say(msg["role"], msg["content"])
        deliverables.append({"format": fmt, **msg})
    return {"deliverables": deliverables}


# ── Graph ──────────────────────────────────────────────────────────────────

workflow = StateGraph(DebateState)
workflow.add_node("cast", cast_node)
workflow.add_node("frame", frame_node)
workflow.add_node("pitch", pitch_node)
workflow.add_node("clash", clash_node)
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("produce", produce_node)

workflow.set_entry_point("cast")
workflow.add_edge("cast", "frame")
workflow.add_edge("frame", "pitch")
workflow.add_conditional_edges("pitch", should_continue,
                               {"clash": "clash", "synthesize": "synthesize"})
workflow.add_conditional_edges("clash", should_continue,
                               {"clash": "clash", "synthesize": "synthesize"})
workflow.add_edge("synthesize", "produce")
workflow.add_edge("produce", END)

app = workflow.compile()
