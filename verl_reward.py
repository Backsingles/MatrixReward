"""Group reward callback for the verl training pipeline."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Callable, Sequence

if __package__:
    from .judge import JudgeFailure, make_implicit_judge, make_judge
    from .judge_client import JudgeClient, JudgeRequestError
    from .reward import Rubric, compute_implicit_reward, compute_reward
else:
    from judge import JudgeFailure, make_implicit_judge, make_judge
    from judge_client import JudgeClient, JudgeRequestError
    from reward import Rubric, compute_implicit_reward, compute_reward


def partition_query_groups(uids: Sequence[Any], workers: int) -> list[list[int]]:
    """Balance whole query groups across workers without splitting a group."""
    if workers < 1:
        raise ValueError("at least one reward worker is required")
    groups: dict[Any, list[int]] = {}
    for index, uid in enumerate(uids):
        groups.setdefault(uid, []).append(index)
    buckets: list[list[int]] = [[] for _ in range(min(workers, len(groups)))]
    for indices in groups.values():
        bucket = min(buckets, key=len)
        bucket.extend(indices)
    return buckets


def final_answer(text: str, thinking_mode: str) -> tuple[str, bool]:
    """Extract the answer and check the policy's thinking-tag contract."""
    if thinking_mode not in {"thinking", "no-thinking"}:
        raise ValueError("thinking_mode must be thinking or no-thinking")
    raw = text.strip()
    opens = list(re.finditer(r"<think>", raw, flags=re.IGNORECASE))
    closes = list(re.finditer(r"</think>", raw, flags=re.IGNORECASE))
    if not raw:
        return "", False
    if thinking_mode == "no-thinking":
        return ("", False) if opens or closes else (raw, True)
    if len(opens) != 1 or len(closes) != 1 or opens[0].start() >= closes[0].start():
        return "", False
    answer = raw[closes[0].end():].strip()
    return answer, bool(answer)


def _query_rubrics(item: dict[str, Any]) -> tuple[str, list[Rubric]]:
    payload = item.get("ground_truth")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("ground_truth must contain query and rubrics")
    query = payload.get("query")
    rows = payload.get("rubrics")
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    if not isinstance(query, str) or not query.strip() or not isinstance(rows, list) or not rows:
        raise ValueError("query and rubrics must be nonempty")
    rubrics = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("rubric"), str) or not row["rubric"].strip():
            raise ValueError("each rubric must have nonempty text")
        points = row.get("points")
        if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points) or points <= 0:
            raise ValueError("rubric points must be positive and finite")
        rubrics.append(Rubric(row["rubric"], float(points)))
    return query, rubrics


def compute_score(
    items: Sequence[dict[str, Any]], *, num_generations: int = 8,
    judge_mode: str = "explicit", judge_workers: int = 16,
    thinking_mode: str = "no-thinking", format_reward_weight: float = 1.0,
    manual_weights: Sequence[float] | None = None,
    request: Callable[[str, str], str] | None = None,
    judge_timeout_seconds: float = 180.0, judge_retries: int = 3,
) -> list[dict[str, float]]:
    """Score all rollouts of one query, including the format-validity term."""
    if num_generations < 2 or len(items) != num_generations:
        raise ValueError("reward callback requires a complete generation group")
    if judge_mode not in {"explicit", "implicit"}:
        raise ValueError("judge_mode must be explicit or implicit")
    if not math.isfinite(format_reward_weight) or format_reward_weight < 0:
        raise ValueError("format_reward_weight must be finite and nonnegative")
    query, rubrics = _query_rubrics(items[0])
    if any(_query_rubrics(item) != (query, rubrics) for item in items[1:]):
        raise ValueError("generation group contains different queries or rubrics")
    parsed = [final_answer(str(item.get("solution_str") or ""), thinking_mode) for item in items]
    answers = [answer for answer, _ in parsed]
    if request is None:
        endpoint = os.environ.get("JUDGE_API_URL", "")
        if not endpoint:
            raise ValueError("JUDGE_API_URL is required")
        request = JudgeClient(
            endpoint, api_key=os.environ.get("JUDGE_API_KEY", ""),
            timeout=judge_timeout_seconds, retries=judge_retries,
            response_format="json_schema" if judge_mode == "explicit" else "json_object",
            max_tokens=512,
        )

    # An invalid final-answer format loses to a valid answer under each rubric.
    # Empty strings here represent invalid outputs, rather than judge inputs.
    if judge_mode == "explicit":
        remote_judge = make_judge(request)

        def judge(q, a, b, rubric):
            if not a or not b:
                return "TIE" if bool(a) == bool(b) else ("A" if a else "B")
            return remote_judge(q, a, b, rubric)

        scorer = compute_reward
    else:
        remote_judge = make_implicit_judge(request)

        def judge(q, a, b, rubric_list):
            if not a or not b:
                verdict = "TIE" if bool(a) == bool(b) else ("A" if a else "B")
                return [verdict] * len(rubric_list)
            return remote_judge(q, a, b, rubric_list)

        scorer = compute_implicit_reward
    try:
        qualities, diagnostics = scorer(
            query, answers, rubrics, judge,
            manual_weights=manual_weights, judge_workers=judge_workers,
        )
    except (JudgeFailure, JudgeRequestError):
        return [{"score": 0.5, "quality_reward": 0.5, "format_reward": 0.0,
                 "format_valid": float(valid), "judge_group_valid": 0.0,
                 "matrix_coefficient": 0.0, "human_coefficient": 0.0,
                 "combination_fallback": 0.0} for _, valid in parsed]
    return [{
        "score": float(quality) + format_reward_weight * float(valid),
        "quality_reward": float(quality), "format_reward": float(valid),
        "format_valid": float(valid), "judge_group_valid": 1.0,
        "matrix_coefficient": float(diagnostics["matrix_coefficient"]),
        "human_coefficient": float(diagnostics["human_coefficient"]),
        "combination_fallback": float(diagnostics["combination_fallback"]),
    } for quality, (_, valid) in zip(qualities, parsed, strict=True)]
