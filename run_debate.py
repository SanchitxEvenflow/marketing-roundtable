"""Run the marketing roundtable.

Examples:
    python run_debate.py "Viral reel + LinkedIn post for a fintech credit card app targeting young professionals"
    python run_debate.py "D2C protein brand launch for Gen Z" --formats reel ad_copy --personas 3 --rounds 1
"""
import argparse
import os
import re
from datetime import datetime

import config
from core.llm import usage_report
from graph.debate_graph import app


def save_output(result: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", result["brief"].lower())[:50].strip("-")
    path = os.path.join(out_dir, f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md")

    lines = [f"# Roundtable: {result['brief']}\n"]
    lines.append(f"**Experts:** {', '.join(p['name'] for p in result['personas'])}\n")

    lines.append("\n## Debate Transcript\n")
    for msg in result["transcript"]:
        lines.append(f"### {msg['role']}\n\n{msg['content']}\n")

    lines.append("\n---\n\n# Creative Strategy Brief\n")
    lines.append(result["creative_brief"] + "\n")

    lines.append("\n---\n\n# Deliverables\n")
    for d in result.get("deliverables", []):
        lines.append(f"\n## {d['role']}\n\n{d['content']}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description="Creative marketing roundtable")
    parser.add_argument("brief", help="The marketing brief")
    parser.add_argument("--formats", nargs="*", default=None,
                        help="Deliverables: reel linkedin ad_copy campaign "
                             "(default: moderator decides from the brief)")
    parser.add_argument("--personas", type=int, default=config.NUM_PERSONAS,
                        help="Experts at the table")
    parser.add_argument("--rounds", type=int, default=config.MAX_CLASH_ROUNDS,
                        help="Max clash rounds after the opening pitches")
    parser.add_argument("--out", default=config.OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    result = app.invoke({
        "brief": args.brief,
        "formats": args.formats or [],
        "num_personas": args.personas,
        "max_rounds": args.rounds,
    })

    path = save_output(result, args.out)
    print(f"\n{'=' * 60}")
    print(f"✅ Full output saved: {path}")
    print(f"📊 Token usage: {usage_report()}")


if __name__ == "__main__":
    main()
