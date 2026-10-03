"""Group reward integration checks without a GPU training runtime."""

import asyncio
import importlib
import json
import sys
import types
import unittest
from unittest import mock

import numpy as np

from generation_prompts import parse_generated_rubrics
from prepare_verl_data import convert_record
from verl_reward import compute_score, final_answer, partition_query_groups


def query_record():
    generated = json.dumps({"rubrics": [
        {"text": "Correctness", "weight": 3},
        {"text": "Completeness", "weight": 2},
        {"text": "Clarity", "weight": 1},
    ]})
    return {"query": "q", "prompt": [{"role": "user", "content": "q"}],
            "rubrics_json": json.dumps(parse_generated_rubrics(generated))}


def group(answers):
    payload = convert_record(query_record())["reward_model"]["ground_truth"]
    return [{"solution_str": answer, "ground_truth": payload} for answer in answers]


def ranked_request(_model, prompt):
    first = prompt.split("Response A:\n", 1)[1].split("\n\nResponse B:", 1)[0]
    verdict = "A" if first == "best" else "B"
    if '"winners"' in prompt:
        return json.dumps({"winners": [verdict] * 3})
    return json.dumps({"winner": verdict})


class GroupRewardTest(unittest.TestCase):
    def test_generation_to_verl_to_reward(self):
        request = mock.Mock(side_effect=ranked_request)
        scores = compute_score(group(["best", "worst"]), num_generations=2,
                               request=request, judge_workers=1)
        np.testing.assert_allclose([row["score"] for row in scores], [2, 1])
        self.assertEqual(request.call_count, 3)  # One request per rubric.
        self.assertTrue(all(row["format_valid"] == 1 for row in scores))

    def test_implicit_requests_single_order(self):
        request = mock.Mock(side_effect=ranked_request)
        scores = compute_score(group(["best", "worst"]), num_generations=2,
                               judge_mode="implicit", request=request, judge_workers=1)
        np.testing.assert_allclose([row["score"] for row in scores], [2, 1])
        self.assertEqual(request.call_count, 1)

    def test_paper_call_counts_for_eight_rollouts_and_four_rubrics(self):
        items = group([f"answer {i}" for i in range(8)])
        items[0]["ground_truth"]["rubrics"].append({"rubric": "Instruction compliance", "points": 3})
        for mode, response, expected in [
            ("explicit", '{"winner":"TIE"}', 112),
            ("implicit", '{"winners":["TIE","TIE","TIE","TIE"]}', 28),
        ]:
            request = mock.Mock(return_value=response)
            scores = compute_score(items, judge_mode=mode, request=request, judge_workers=4)
            self.assertEqual(request.call_count, expected)
            np.testing.assert_allclose([row["score"] for row in scores], [1.5] * 8)

    def test_invalid_format_loses_without_judge_request(self):
        request = mock.Mock(side_effect=AssertionError("unexpected judge call"))
        scores = compute_score(group(["best", "<think>hidden</think>worst"]),
                               num_generations=2, request=request)
        np.testing.assert_allclose([row["score"] for row in scores], [2, 0])
        request.assert_not_called()

    def test_both_invalid_tie(self):
        scores = compute_score(group(["", "<think>hidden</think>"]), num_generations=2,
                               request=mock.Mock(side_effect=AssertionError))
        np.testing.assert_allclose([row["score"] for row in scores], [.5, .5])

    def test_thinking_only_final_answer_is_judged(self):
        request = mock.Mock(side_effect=ranked_request)
        compute_score(group(["<think>hidden</think>best", "<think>hidden</think>worst"]),
                      num_generations=2, thinking_mode="thinking", request=request, judge_workers=1)
        for call in request.call_args_list:
            self.assertNotIn("hidden", call.args[1])
            self.assertNotIn("<think>", call.args[1])

    def test_failed_judge_neutralizes_entire_group(self):
        request = mock.Mock(return_value="invalid")
        scores = compute_score(group(["best", "worst", ""]), num_generations=3,
                               request=request, judge_workers=1)
        np.testing.assert_allclose([row["score"] for row in scores], [.5, .5, .5])
        self.assertTrue(all(row["judge_group_valid"] == 0 for row in scores))
        self.assertTrue(all(row["format_reward"] == 0 for row in scores))
        self.assertEqual(request.call_count, 3)

    def test_rejects_incomplete_and_mixed_groups(self):
        with self.assertRaises(ValueError):
            compute_score(group(["best"]), num_generations=2)
        items = group(["best", "worst"])
        items[1]["ground_truth"] = dict(items[1]["ground_truth"], query="different")
        with self.assertRaises(ValueError):
            compute_score(items, num_generations=2)

    def test_tag_contract(self):
        self.assertEqual(final_answer("<THINK>trace</THINK>final", "thinking"), ("final", True))
        self.assertEqual(final_answer("<think>trace</think>", "thinking"), ("", False))
        self.assertEqual(final_answer("<think>x</think><think>y</think>final", "thinking"), ("", False))
        self.assertEqual(final_answer("final", "no-thinking"), ("final", True))

    def test_interleaved_group_partition(self):
        uids = ["q1", "q2", "q1", "q3", "q2", "q3"]
        buckets = partition_query_groups(uids, 2)
        self.assertEqual(sorted(i for bucket in buckets for i in bucket), list(range(6)))
        for uid in set(uids):
            self.assertEqual(sum(any(uids[i] == uid for i in bucket) for bucket in buckets), 1)
        self.assertEqual(len(partition_query_groups(uids, 8)), 3)
        self.assertEqual(partition_query_groups([], 8), [])

    def test_converter_rejects_wrong_prompt_or_weights(self):
        row = query_record()
        with self.assertRaises(ValueError):
            convert_record(dict(row, prompt=[]))
        rubrics = json.loads(row["rubrics_json"])
        rubrics[0]["points"] = .5
        with self.assertRaises(ValueError):
            convert_record(dict(row, rubrics_json=json.dumps(rubrics)))


class Batch:
    def __init__(self):
        self.batch = {"prompts": np.array([[1], [2], [1], [2]]),
                      "responses": np.arange(4).reshape(4, 1),
                      "attention_mask": np.ones((4, 2), dtype=int)}
        self.non_tensor_batch = {"uid": np.array(["one", "two", "one", "two"]),
                                 "reward_model": [convert_record(query_record())["reward_model"]] * 4}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return types.SimpleNamespace(batch={key: value[index] for key, value in self.batch.items()},
                                     non_tensor_batch={key: value[index] for key, value in self.non_tensor_batch.items()})


def manager_class():
    # Replace only the heavy framework base; execute the real integration class.
    base_module = types.ModuleType("verl.experimental.reward_loop.reward_manager.base")

    class Base:
        def __init__(self, config, tokenizer, compute_score):
            self.tokenizer = tokenizer
            self.compute_score = compute_score
            self.loop = asyncio.get_running_loop()

    base_module.RewardManagerBase = Base
    name = "verl_reward_manager"
    with mock.patch.dict(sys.modules, {base_module.__name__: base_module}):
        sys.modules.pop(name, None)
        return importlib.import_module(name).MatrixRewardManager


class RewardManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_callback_gets_whole_groups_and_order_is_restored(self):
        seen = []

        def callback(items):
            values = [int(item["solution_str"]) for item in items]
            seen.append(values)
            return [{"score": value / 10} for value in values]

        manager = manager_class()(None, types.SimpleNamespace(decode=lambda ids, **_: str(ids[0])), callback)
        results = await manager.run_batch(Batch())
        self.assertEqual(seen, [[0, 2], [1, 3]])
        self.assertEqual([row["reward_score"] for row in results], [0, .1, .2, .3])
        self.assertEqual(manager.group_indices(Batch(), 2), [[0, 2], [1, 3]])

    async def test_rejects_mixed_prompts_and_per_answer_callback(self):
        manager = manager_class()(None, None, lambda items: [])
        batch = Batch()
        batch.batch["prompts"][2, 0] = 9
        with self.assertRaises(ValueError):
            manager.group_indices(batch, 2)
        with self.assertRaises(ValueError):
            await manager.run_single(batch[0])


if __name__ == "__main__":
    unittest.main()
