"""Small, dependency-free helpers for DAPO dynamic sampling and length shaping."""

from collections import OrderedDict
import math


def informative_group_indices(uids, scores, group_size, valid=None):
    """Return complete, valid groups whose final rewards are not all equal."""
    if len(uids) != len(scores) or (valid is not None and len(valid) != len(uids)):
        raise ValueError("DAPO group fields must have equal lengths")
    groups = OrderedDict()
    for index, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(index)
    kept = []
    for indices in groups.values():
        if len(indices) != group_size:
            raise ValueError(f"DAPO requires {group_size} rollouts per query; got {len(indices)}")
        if valid is not None and not all(bool(valid[i]) for i in indices):
            continue
        values = [float(scores[i]) for i in indices]
        if not all(math.isfinite(value) for value in values):
            continue
        if min(values) != max(values):
            kept.extend(indices)
    return kept


def overlong_penalties(lengths, max_response_length, buffer_length, penalty_factor=1.0):
    """DAPO's linear penalty in the final response-length buffer."""
    if not 0 < buffer_length <= max_response_length or not math.isfinite(penalty_factor) or penalty_factor < 0:
        raise ValueError("Invalid DAPO overlong buffer or penalty factor")
    expected = max_response_length - buffer_length
    return [min(-(int(length) - expected) / buffer_length * penalty_factor, 0.0) for length in lengths]
