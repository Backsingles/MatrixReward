"""Render the query-specific rubric generation prompt pair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


PROMPTS = Path(__file__).parent / "prompts"


def generation_messages(
    query: str, candidates: Sequence[tuple[float, str]],
) -> list[dict[str, str]]:
    """Format four responses by descending overall score out of ten."""
    if len(candidates) != 4:
        raise ValueError("exactly four candidate answers are required")
    if any(not 0 <= score <= 10 for score, _ in candidates):
        raise ValueError("overall scores must be between zero and ten")
    answers = "\n\n".join(
        f"Rank {rank}; overall score {score:g}/10:\n{answer}"
        for rank, (score, answer) in enumerate(
            sorted(candidates, key=lambda item: item[0], reverse=True), 1
        )
    )
    system = (PROMPTS / "rubric_generation_system.txt").read_text(encoding="utf-8").rstrip("\n")
    user = (PROMPTS / "rubric_generation_user.txt").read_text(encoding="utf-8").format(
        query=query, answers=answers,
    ).rstrip("\n")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_generated_rubrics(response: str | dict[str, Any]) -> list[dict[str, str | int]]:
    """Convert the generation prompt's JSON output to the training rubric schema."""
    payload = json.loads(response) if isinstance(response, str) else response
    if not isinstance(payload, dict) or set(payload) != {"rubrics"}:
        raise ValueError("generated response must contain only rubrics")
    items = payload["rubrics"]
    if not isinstance(items, list) or not 3 <= len(items) <= 5:
        raise ValueError("generated response must contain 3-5 rubrics")
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"text", "weight"}:
            raise ValueError("generated rubric must contain text and weight")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("generated rubric text must be nonempty")
        if type(item["weight"]) is not int or item["weight"] not in (1, 2, 3):
            raise ValueError("generated rubric weight must be an integer from 1 to 3")
        result.append({"rubric": item["text"].strip(), "points": item["weight"]})
    return result
