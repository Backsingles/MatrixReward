"""Exercise the bundled framework's group reward and GRPO code on CPU stand-ins."""

import ast
import asyncio
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import types
import unittest

import numpy as np

from tests.test_verl import manager_class, query_record, ranked_request
from prepare_verl_data import convert_record
from verl_reward import compute_score

FRAMEWORK = Path(__file__).resolve().parents[1] / "third_party" / "verl" / "verl"


def load_function(relative_path, name, namespace, class_name=None):
    tree = ast.parse((FRAMEWORK / relative_path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative_path, "exec"), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, values, dtype=None):
        self.values = np.asarray(values, dtype=dtype)

    @property
    def shape(self):
        return self.values.shape

    def size(self, dim):
        return self.shape[dim]

    def __len__(self):
        return len(self.values)

    def __int__(self):
        return int(self.values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __getitem__(self, key):
        result = self.values[key]
        return Tensor(result) if isinstance(result, np.ndarray) else result

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = tuple(np.asarray(k) if isinstance(k, Tensor) else k for k in key)
        self.values[key] = np.asarray(value)

    def sum(self, dim=None):
        return Tensor(self.values.sum(axis=dim))

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.values, dim))

    def __sub__(self, other):
        return Tensor(self.values - np.asarray(other))

    def __mul__(self, other):
        return Tensor(self.values * np.asarray(other))


TORCH = types.SimpleNamespace(
    tensor=Tensor, zeros_like=lambda value, **_: Tensor(np.zeros_like(np.asarray(value), dtype=float)),
    arange=np.arange, float32=np.float32, equal=lambda a, b: np.array_equal(a, b),
    stack=lambda values: Tensor(np.stack(values)), mean=lambda value: np.mean(np.asarray(value)),
    std=lambda value: np.std(np.asarray(value), ddof=1), no_grad=nullcontext,
)


class Data:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch if batch is not None else {
            "prompts": Tensor([[1], [2], [3], [1], [2], [3]]),
            "responses": Tensor([[0], [0], [0], [1], [1], [1]]),
            "attention_mask": Tensor(np.ones((6, 2), dtype=int)),
        }
        self.non_tensor_batch = non_tensor_batch if non_tensor_batch is not None else {
            "uid": np.array(["q1", "q2", "q3", "q1", "q2", "q3"]),
            "reward_model": np.array([convert_record(query_record())["reward_model"]] * 6, dtype=object),
        }
        self.meta_info = meta_info or {}

    def __len__(self):
        return len(next(iter(self.batch.values())))

    def __getitem__(self, index):
        return types.SimpleNamespace(batch={k: v[index] for k, v in self.batch.items()},
                                     non_tensor_batch={k: v[index] for k, v in self.non_tensor_batch.items()})

    def select_idxs(self, indices):
        return Data({k: v[indices] for k, v in self.batch.items()},
                    {k: v[indices] for k, v in self.non_tensor_batch.items()})


class BundledFrameworkTest(unittest.TestCase):
    def test_native_worker_scheduling_and_grpo_advantages(self):
        relative = "experimental/reward_loop/reward_loop.py"
        worker_method = load_function(relative, "compute_score_batch", {"asyncio": asyncio}, "RewardLoopWorker")
        called = []

        async def run_batch(data):
            manager = manager_class()(
                None, types.SimpleNamespace(decode=lambda ids, **_: "best" if ids[0] == 0 else "worst"),
                lambda items: compute_score(items, num_generations=2, request=ranked_request, judge_workers=1),
            )
            called.append(list(data.non_tensor_batch["uid"]))
            return await worker_method(types.SimpleNamespace(reward_manager=manager), data)

        namespace = {"np": np, "torch": TORCH, "json": json,
                     "ray": types.SimpleNamespace(get=lambda requests: requests),
                     "TensorDict": lambda batch, **_: batch, "DataProto": Data}
        chunk = load_function(relative, "_chunk_by_uid", namespace, "RewardLoopManager")
        compute = load_function(relative, "compute_rm_score", namespace, "RewardLoopManager")
        workers = [types.SimpleNamespace(compute_score_batch=types.SimpleNamespace(
            remote=lambda data: asyncio.run(run_batch(data)))) for _ in range(5)]
        manager = types.SimpleNamespace(
            config=types.SimpleNamespace(reward=types.SimpleNamespace(reward_manager=types.SimpleNamespace(name="MatrixRewardManager"))),
            reward_model_manager=None, reward_manager_cls=types.SimpleNamespace(requires_grouped_rollouts=True),
            reward_loop_workers=workers,
        )
        manager._chunk_by_uid = types.MethodType(chunk, manager)
        data = Data()
        result = compute(manager, data)
        np.testing.assert_allclose(result.batch["rm_scores"], [[2], [2], [2], [1], [1], [1]])
        self.assertEqual(called, [["q1", "q1"], ["q2", "q2"], ["q3", "q3"]])
        self.assertEqual(len(result.non_tensor_batch["format_valid"]), 6)
        grpo = load_function("trainer/ppo/core_algos.py", "compute_grpo_outcome_advantage",
                             {"torch": TORCH, "defaultdict": defaultdict})
        advantages, _ = grpo(result.batch["rm_scores"], Tensor(np.ones((6, 1))), data.non_tensor_batch["uid"])
        expected = .5 / (np.std([2., 1.], ddof=1) + 1e-6)
        np.testing.assert_allclose(advantages, [[expected]] * 3 + [[-expected]] * 3)
        neutral, _ = grpo(Tensor(np.ones((6, 1)) * .5), Tensor(np.ones((6, 1))), data.non_tensor_batch["uid"])
        np.testing.assert_allclose(neutral, np.zeros((6, 1)))

    def test_validation_removes_padding_before_group_reward(self):
        namespace = {}
        load_function("trainer/ppo/ray_trainer.py", "_validate", namespace, "RayPPOTrainer")
        tree = ast.parse((FRAMEWORK / "trainer/ppo/ray_trainer.py").read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        function = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_validate")
        source = ast.unparse(function)
        self.assertLess(source.index("unpad_dataproto("), source.index("self._compute_reward_colocate(test_output_gen_batch)"))


if __name__ == "__main__":
    unittest.main()
