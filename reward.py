"""Rubric-wise comparison, matrix weighting, and reward calculation."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from random import getrandbits
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Rubric:
    rubric: str
    points: float


Judge = Callable[[str, str, str, Rubric], str]
ImplicitJudge = Callable[[str, str, str, Sequence[Rubric]], Sequence[str]]


def _verdict(value: str) -> float:
    verdict = value.strip().upper()
    if verdict not in {"A", "B", "TIE"}:
        raise ValueError(f"judge must return A, B, or TIE; got {value!r}")
    return {"A": 1.0, "B": 0.0, "TIE": 0.5}[verdict]


def build_rubricwise_matrix(
    query: str, answers: Sequence[str], rubrics: Sequence[Rubric], judge: Judge,
    *, judge_workers: int = 1,
) -> np.ndarray:
    """Judge each unordered pair once per rubric, in a random display order."""
    if not answers or not rubrics:
        raise ValueError("at least one answer and rubric are required")
    if judge_workers < 1:
        raise ValueError("judge_workers must be positive")
    matrix = np.zeros((len(answers), len(rubrics)), dtype=float)
    tasks = [(i, j, k, rubric, bool(getrandbits(1)))
             for i, j in combinations(range(len(answers)), 2)
             for k, rubric in enumerate(rubrics)]

    def compare(task: tuple[int, int, int, Rubric, bool]) -> tuple[int, int, int, float]:
        i, j, k, rubric, swapped = task
        first, second = (j, i) if swapped else (i, j)
        verdict = _verdict(judge(query, answers[first], answers[second], rubric))
        return i, j, k, 1.0 - verdict if swapped else verdict

    if judge_workers == 1:
        results = map(compare, tasks)
    else:
        pool = ThreadPoolExecutor(max_workers=judge_workers)
        results = pool.map(compare, tasks)
    try:
        for i, j, k, win_i in results:
            matrix[i, k] += win_i
            matrix[j, k] += 1.0 - win_i
    finally:
        if judge_workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)
    if len(answers) == 1:
        return np.full_like(matrix, 0.5)
    return matrix / (len(answers) - 1)


def build_implicit_rubricwise_matrix(
    query: str, answers: Sequence[str], rubrics: Sequence[Rubric],
    judge: ImplicitJudge, *, judge_workers: int = 1,
) -> np.ndarray:
    """Judge each pair once for all rubrics, in a random display order."""
    if not answers or not rubrics:
        raise ValueError("at least one answer and rubric are required")
    if judge_workers < 1:
        raise ValueError("judge_workers must be positive")
    matrix = np.zeros((len(answers), len(rubrics)), dtype=float)
    tasks = [(i, j, bool(getrandbits(1))) for i, j in combinations(range(len(answers)), 2)]

    def compare(pair: tuple[int, int, bool]) -> tuple[int, int, np.ndarray]:
        i, j, swapped = pair
        first, second = (j, i) if swapped else (i, j)
        verdicts = judge(query, answers[first], answers[second], rubrics)
        if len(verdicts) != len(rubrics):
            raise ValueError("judge returned a different number of rubric verdicts")
        wins = np.array([_verdict(verdict) for verdict in verdicts])
        return i, j, 1.0 - wins if swapped else wins

    if judge_workers == 1:
        results = map(compare, tasks)
    else:
        pool = ThreadPoolExecutor(max_workers=judge_workers)
        results = pool.map(compare, tasks)
    try:
        for i, j, wins in results:
            matrix[i] += wins
            matrix[j] += 1.0 - wins
    finally:
        if judge_workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)
    if len(answers) == 1:
        return np.full_like(matrix, 0.5)
    return matrix / (len(answers) - 1)


def extract_matrix_weights(matrix: np.ndarray) -> np.ndarray:
    """Extract rubric weights from column contrast and correlation conflict.

    Constant columns get zero weight unless every column is constant; that
    case uses uniform weights. Undefined correlations are treated as zero.
    """
    values = _matrix(matrix)
    rows, cols = values.shape
    if cols == 0:
        raise ValueError("matrix needs at least one rubric column")
    if rows < 2:
        return np.full(cols, 1.0 / cols)
    low, high = values.min(axis=0), values.max(axis=0)
    live = (high - low) > 1e-12
    normalized = np.zeros_like(values)
    normalized[:, live] = (values[:, live] - low[live]) / (high[live] - low[live])
    sigma = normalized.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.atleast_2d(np.corrcoef(normalized, rowvar=False))
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    conflict = (1.0 - correlation).sum(axis=1)
    information = sigma * conflict
    total = float(information.sum())
    return information / total if total > 1e-12 else np.full(cols, 1.0 / cols)


def _matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be a finite 2D array")
    return values


def spatial_distance_rewards(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vector-normalized closeness to the best and worst rubric profiles."""
    values = _matrix(matrix)
    weights = np.asarray(weights, dtype=float)
    if values.shape[1] == 0 or weights.shape != (values.shape[1],):
        raise ValueError("one weight is required per rubric column")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and sum positive")
    weights = weights / weights.sum()
    if values.shape[0] == 0:
        return np.zeros(0)
    norm = np.linalg.norm(values, axis=0)
    normalized = np.divide(values, norm, out=np.zeros_like(values), where=norm > 1e-12)
    weighted = normalized * weights
    positive, negative = weighted.max(axis=0), weighted.min(axis=0)
    d_plus = np.linalg.norm(weighted - positive, axis=1)
    d_minus = np.linalg.norm(weighted - negative, axis=1)
    denominator = d_plus + d_minus
    return np.divide(d_minus, denominator, out=np.full(values.shape[0], 0.5), where=denominator > 1e-12)


def combine_weights(
    matrix_weights: np.ndarray, human_weights: np.ndarray,
    *, diagnostics: dict[str, float] | None = None,
) -> np.ndarray:
    """Combine two normalized weight vectors through a least-squares system.

    If the vectors are parallel, use their normalized geometric mean. If that
    has zero mass, use the human weights.
    """
    extracted = np.asarray(matrix_weights, dtype=float)
    human = np.asarray(human_weights, dtype=float)
    if (extracted.ndim != 1 or extracted.shape != human.shape or extracted.size == 0
            or not np.all(np.isfinite(extracted)) or not np.all(np.isfinite(human))
            or np.any(extracted < 0) or np.any(human < 0)
            or extracted.sum() <= 0 or human.sum() <= 0):
        raise ValueError("weight vectors must have matching positive, finite mass")
    extracted = extracted / extracted.sum()
    human = human / human.sum()
    gram = np.array([
        [extracted @ extracted, extracted @ human],
        [human @ extracted, human @ human],
    ])
    rhs = np.array([extracted @ extracted, human @ human])
    coefficients = None
    if abs(float(np.linalg.det(gram))) > 1e-12:
        try:
            solved = np.linalg.solve(gram, rhs)
            if np.all(np.isfinite(solved)):
                coefficients = np.abs(solved)
        except np.linalg.LinAlgError:
            pass
    fallback = coefficients is None or float(coefficients.sum()) <= 1e-12
    if fallback:
        geometric = np.sqrt(extracted * human)
        total = float(geometric.sum())
        combined = geometric / total if total > 1e-12 else human
        coefficients = np.array([0.5, 0.5])
    else:
        coefficients = coefficients / float(coefficients.sum())
        combined = coefficients[0] * extracted + coefficients[1] * human
    if diagnostics is not None:
        diagnostics.update({
            "matrix_coefficient": float(coefficients[0]),
            "human_coefficient": float(coefficients[1]),
            "combination_fallback": float(fallback),
        })
    return combined


def compute_reward(
    query: str, answers: Sequence[str], rubrics: Sequence[Rubric], judge: Judge,
    *, manual_weights: Sequence[float] | None = None, judge_workers: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Build the matrix, combine extracted and human weights, then score.

    Rubric points are used as human weights unless manual_weights is supplied
    in the same rubric order.
    """
    matrix = build_rubricwise_matrix(query, answers, rubrics, judge, judge_workers=judge_workers)
    return _score_matrix(matrix, rubrics, manual_weights)


def compute_implicit_reward(
    query: str, answers: Sequence[str], rubrics: Sequence[Rubric],
    judge: ImplicitJudge, *, manual_weights: Sequence[float] | None = None,
    judge_workers: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    matrix = build_implicit_rubricwise_matrix(
        query, answers, rubrics, judge, judge_workers=judge_workers,
    )
    return _score_matrix(matrix, rubrics, manual_weights)


def _score_matrix(
    matrix: np.ndarray, rubrics: Sequence[Rubric],
    manual_weights: Sequence[float] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    learned = extract_matrix_weights(matrix)
    objective = np.asarray(
        [rubric.points for rubric in rubrics] if manual_weights is None else manual_weights,
        dtype=float,
    )
    if objective.shape != (len(rubrics),) or not np.all(np.isfinite(objective)) or np.any(objective < 0) or objective.sum() <= 0:
        raise ValueError("human weights must match rubrics, be finite and nonnegative, and sum positive")
    objective = objective / objective.sum()
    combination_diagnostics: dict[str, float] = {}
    combined = combine_weights(learned, objective, diagnostics=combination_diagnostics)
    return spatial_distance_rewards(matrix, combined), {
        "matrix": matrix, "matrix_weights": learned,
        "human_weights": objective, "combined_weights": combined,
        **combination_diagnostics,
    }
