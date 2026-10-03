"""Convert supplied query-specific JSONL records to verl parquet records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_record(row: dict[str, Any]) -> dict[str, Any]:
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be nonempty text")
    prompt = [{"role": "user", "content": query}]
    if row.get("prompt") != prompt:
        raise ValueError("prompt must contain the matching user query")
    rubrics = json.loads(row["rubrics_json"])
    if not isinstance(rubrics, list) or not 3 <= len(rubrics) <= 5:
        raise ValueError("each query requires 3-5 rubrics")
    for rubric in rubrics:
        if (not isinstance(rubric, dict) or set(rubric) != {"rubric", "points"}
                or not isinstance(rubric["rubric"], str) or not rubric["rubric"].strip()
                or type(rubric["points"]) is not int or rubric["points"] not in {1, 2, 3}):
            raise ValueError("rubrics must contain text and integer points from 1 to 3")
    return {"data_source": "query_specific", "prompt": prompt,
            "reward_model": {"style": "rubric", "ground_truth": {"query": query, "rubrics": rubrics}}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("--output is required unless --check is used")
    rows = [convert_record(json.loads(line)) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input file contains no query records")
    if args.check:
        print(f"Validated {len(rows)} query-specific records")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(f"Converted {len(rows)} records")


if __name__ == "__main__":
    main()
