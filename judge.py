"""Single-rubric judge prompt and strict response parser for MatrixReward."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Callable, Sequence

if __package__:
    from .reward import Rubric
else:
    from reward import Rubric

JUDGE_MODEL_NAME = "qwen36_35b_a3b"


@lru_cache(maxsize=1)
def _prompt_template() -> Template:
    path = Path(__file__).parent / "prompts" / "judge_single_rubric.txt"
    return Template(path.read_text(encoding="utf-8").rstrip("\n"))


@lru_cache(maxsize=1)
def _all_rubrics_template() -> Template:
    path = Path(__file__).parent / "prompts" / "judge_all_rubrics.txt"
    return Template(path.read_text(encoding="utf-8").rstrip("\n"))


class JudgeFailure(RuntimeError):
    """The judge did not return a usable verdict after three attempts."""


def dimension_prompt(query: str, answer_a: str, answer_b: str, rubric: Rubric) -> str:
    """Construct a single-rubric comparison prompt."""
    return _prompt_template().substitute(
        rubrics=rubric.rubric, query=query,
        response_a=answer_a, response_b=answer_b,
    )


def all_dimensions_prompt(
    query: str, answer_a: str, answer_b: str, rubrics: Sequence[Rubric],
) -> str:
    """Construct the one-request, all-rubric comparison prompt."""
    if not rubrics:
        raise ValueError("at least one rubric is required")
    return _all_rubrics_template().substitute(
        K=len(rubrics), query=query, response_a=answer_a, response_b=answer_b,
        numbered_rubrics="\n".join(
            f"{index}. {rubric.rubric}" for index, rubric in enumerate(rubrics, 1)
        ),
    )


def parse_verdict(response: str) -> str:
    value = json.loads(response)
    if (not isinstance(value, dict) or set(value) != {"winner"}
            or not isinstance(value["winner"], str)
            or value["winner"] not in {"A", "B", "TIE"}):
        raise ValueError("judge response must be JSON with winner A, B, or TIE")
    return value["winner"]


def parse_verdicts(response: str, count: int) -> list[str]:
    value = json.loads(response)
    if (not isinstance(value, dict) or set(value) != {"winners"}
            or not isinstance(value["winners"], list)
            or len(value["winners"]) != count
            or any(not isinstance(item, str) or item not in {"A", "B", "TIE"}
                   for item in value["winners"])):
        raise ValueError("judge response must have one A, B, or TIE verdict per rubric")
    return value["winners"]


def _valid_response(request: Callable[[str, str], str], model_name: str,
                    prompt: str, parse: Callable[[str], object]) -> object:
    for attempt in range(3):
        try:
            return parse(request(model_name, prompt))
        except (ValueError, json.JSONDecodeError) as error:
            if attempt == 2:
                raise JudgeFailure("judge returned invalid verdicts three times") from error
    raise AssertionError("unreachable")


def make_judge(
    request: Callable[[str, str], str], *, model_name: str = JUDGE_MODEL_NAME,
):
    """Adapt a model-and-prompt request function to compute_reward."""
    def judge(query: str, answer_a: str, answer_b: str, rubric: Rubric) -> str:
        return _valid_response(request, model_name,
                               dimension_prompt(query, answer_a, answer_b, rubric),
                               parse_verdict)
    return judge


def make_implicit_judge(
    request: Callable[[str, str], str], *, model_name: str = JUDGE_MODEL_NAME,
):
    """Adapt a request function to one verdict per rubric in one response."""
    def judge(query: str, answer_a: str, answer_b: str,
              rubrics: Sequence[Rubric]) -> list[str]:
        return _valid_response(
            request, model_name,
            all_dimensions_prompt(query, answer_a, answer_b, rubrics),
            lambda response: parse_verdicts(response, len(rubrics)),
        )
    return judge
