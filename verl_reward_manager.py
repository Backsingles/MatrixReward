"""verl reward manager that evaluates complete query groups."""

from __future__ import annotations

import inspect

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

if __package__:
    from .verl_reward import partition_query_groups
else:
    from verl_reward import partition_query_groups


class MatrixRewardManager(RewardManagerBase):
    requires_grouped_rollouts = True

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        if compute_score is None:
            raise ValueError("MatrixRewardManager requires a group reward callback")
        super().__init__(config, tokenizer, compute_score)

    @staticmethod
    def group_indices(data, workers):
        if "uid" not in data.non_tensor_batch:
            raise ValueError("query grouping requires verl uid values")
        uids = data.non_tensor_batch["uid"]
        groups = partition_query_groups(uids, len(uids) or 1)
        prompts = data.batch["prompts"]
        for indices in groups:
            if any(not bool((prompts[indices[0]] == prompts[i]).all()) for i in indices[1:]):
                raise ValueError("a query group contains different tokenized prompts")
        return partition_query_groups(uids, workers)

    async def run_single(self, data):
        raise ValueError("MatrixRewardManager needs the grouped reward-loop patch")

    async def run_batch(self, data):
        uids = data.non_tensor_batch["uid"]
        groups = partition_query_groups(uids, len(uids) or 1)
        outputs = [None] * len(data)
        for indices in groups:
            items = []
            for index in indices:
                row = data[index]
                response = row.batch["responses"]
                length = int(row.batch["attention_mask"][-len(response):].sum())
                text = await self.loop.run_in_executor(
                    None, lambda ids=response[:length]: self.tokenizer.decode(ids, skip_special_tokens=True),
                )
                reward_model = row.non_tensor_batch["reward_model"]
                items.append({"solution_str": text, "ground_truth": reward_model["ground_truth"]})
            if inspect.iscoroutinefunction(self.compute_score):
                results = await self.compute_score(items=items)
            else:
                results = await self.loop.run_in_executor(
                    None, lambda group=items: self.compute_score(items=group),
                )
            if len(results) != len(indices):
                raise ValueError("group reward callback returned the wrong number of scores")
            for index, result in zip(indices, results, strict=True):
                outputs[index] = {"reward_score": float(result["score"]), "reward_extra_info": result}
        return outputs
