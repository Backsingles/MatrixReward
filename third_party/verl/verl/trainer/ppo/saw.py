"""SAW Algorithm 1 coefficient-of-variation weights and query adaptation.

Paper: https://arxiv.org/abs/2606.07705
This module contains only NumPy so the numerical rule can be checked without
loading the training stack.
"""

from __future__ import annotations

import numpy as np


def saw_batch_weights(
    rewards: np.ndarray,
    valid_rows: np.ndarray,
    active: np.ndarray,
    minima: np.ndarray,
    delta: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-dimension CV and SAW's GRPO/GDPO weights."""
    if rewards.ndim != 2 or not np.all(np.isfinite(rewards)):
        raise ValueError("SAW requires a finite rollout-by-reward matrix")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("SAW delta must be finite and positive")
    if mode not in {"reward", "advantage"}:
        raise ValueError("SAW mode must be reward or advantage")
    if valid_rows.shape != (rewards.shape[0],) or active.shape != (rewards.shape[1],) or minima.shape != (rewards.shape[1],):
        raise ValueError("SAW row mask, column mask and minima must match rewards")
    if not np.all(np.isfinite(minima)):
        raise ValueError("SAW minima must be finite")
    cv = np.zeros(rewards.shape[1], dtype=np.float64)
    weights = np.zeros_like(cv)
    count = int(active.sum())
    if count == 0 or not valid_rows.any():
        return cv, weights
    shifted = rewards[valid_rows][:, active] - minima[active] + delta
    if np.any(shifted <= 0):
        raise ValueError("SAW reward is below its configured theoretical minimum")
    # The released ToolRL implementation uses torch.std's sample correction.
    # A one-row batch has no measurable dispersion, so take the paper fallback.
    std = shifted.std(axis=0, ddof=1) if shifted.shape[0] > 1 else np.zeros(count)
    cv[active] = std / (shifted.mean(axis=0) + delta)
    total = float(cv.sum())
    if total < delta:
        weights[active] = 1.0
    else:
        weights[active] = cv[active] / total
        if mode == "advantage":
            weights[active] *= count
    return cv, weights


def saw_partition_weights(
    rewards: np.ndarray,
    index: np.ndarray,
    valid_rows: np.ndarray,
    presence: np.ndarray,
    priorities: np.ndarray,
    minima: np.ndarray,
    delta: float,
    mode: str,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute paper SAW over one batch or the local query-specific variant."""
    if scope not in {"batch", "query"}:
        raise ValueError("SAW scope must be batch or query")
    if (index.shape != (rewards.shape[0],) or presence.shape != rewards.shape
            or priorities.shape != rewards.shape):
        raise ValueError("SAW query, presence and priority shapes must match rewards")
    if not np.all(np.isfinite(priorities)) or np.any(priorities < 0):
        raise ValueError("SAW priorities must be finite and nonnegative")
    cv = np.zeros_like(rewards, dtype=np.float64)
    dynamic = np.zeros_like(rewards, dtype=np.float64)
    partitions = [np.arange(rewards.shape[0])] if scope == "batch" else [
        np.flatnonzero(index == group_id) for group_id in np.unique(index)
    ]
    for positions in partitions:
        if positions.size == 0:
            continue
        if not np.all(presence[positions] == presence[positions[0]]):
            if scope == "batch":
                raise ValueError("SAW needs stable reward dimensions across the optimizer batch")
            raise ValueError("Query SAW needs stable rubric presence within each query")
        if not np.allclose(priorities[positions], priorities[positions[0]]):
            if scope == "batch":
                raise ValueError("Batch SAW needs stable priorities across the optimizer batch")
            raise ValueError("Query SAW needs stable rubric priorities within each query")
        active = presence[positions[0]] & (priorities[positions[0]] > 0)
        if not active.any():
            raise ValueError("SAW requires a present reward with positive priority")
        group_cv, group_dynamic = saw_batch_weights(
            rewards[positions], valid_rows[positions], active, minima, delta, mode,
        )
        cv[positions] = group_cv
        dynamic[positions] = group_dynamic
    return cv, dynamic, dynamic * priorities
