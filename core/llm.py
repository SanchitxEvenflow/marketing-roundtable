"""Single gateway to the NVIDIA NIM endpoint (Nemotron-3-Super-120B-A12B).

Every LLM call in the system goes through `chat()` so we get, in one place:
  - retry with backoff on rate limits
  - per-role output token caps (the main lever for token savings)
  - Nemotron thinking mode via enable_thinking=True
  - a running usage tally printed at the end of a run
"""
import json
import re
import time

from langchain_openai import ChatOpenAI

import config

_usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def _make_llm(model: str, max_tokens: int, temperature: float) -> ChatOpenAI:
    if not config.NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com "
            "and add NVIDIA_API_KEY=nvapi-... to your .env"
        )
    return ChatOpenAI(
        base_url=config.NVIDIA_BASE_URL,
        api_key=config.NVIDIA_API_KEY,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=0,  # we handle retries ourselves for clearer backoff
        # Nemotron-3-Super-120B-A12B thinking mode — exposes chain-of-thought.
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        },
    )


def chat(system: str, user: str, tier: str = "large",
         max_tokens: int = 1024, temperature: float = 0.8) -> str:
    """One chat completion. tier='large' for creative voices, 'small' for bookkeeping."""
    model = config.MODEL_LARGE if tier == "large" else config.MODEL_SMALL
    llm = _make_llm(model, max_tokens, temperature)
    messages = [("system", system), ("human", user)]

    last_err = None
    for attempt in range(4):
        try:
            if config.REQUEST_DELAY:
                time.sleep(config.REQUEST_DELAY)
            resp = llm.invoke(messages)
            usage = resp.response_metadata.get("token_usage") or {}
            _usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            _usage["completion_tokens"] += usage.get("completion_tokens", 0)
            _usage["calls"] += 1
            return resp.content.strip()
        except Exception as e:  # 429s surface as generic API errors
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  [llm] {type(e).__name__}: retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def chat_json(system: str, user: str, tier: str = "small",
              max_tokens: int = 1024, temperature: float = 0.3) -> dict:
    """Chat call that must return a JSON object. Parses leniently."""
    raw = chat(system + "\nRespond with a single valid JSON object and nothing else.",
               user, tier=tier, max_tokens=max_tokens, temperature=temperature)
    return parse_json(raw)


def parse_json(raw: str) -> dict:
    text = raw.strip()
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse JSON from model output:\n{raw[:500]}")


def usage_report() -> str:
    total = _usage["prompt_tokens"] + _usage["completion_tokens"]
    return (f"{_usage['calls']} LLM calls | "
            f"{_usage['prompt_tokens']:,} in / {_usage['completion_tokens']:,} out "
            f"({total:,} total tokens)")
