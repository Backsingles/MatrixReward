# verl

This directory contains the verl runtime used by MatrixReward, based on version
0.8.0.dev. It includes training, rollout, distributed workers, configuration,
checkpoint handling, and model export.

Install from the MatrixReward repository root with `bash setup_verl.sh`.
MatrixReward uses the GRPO trainer and the vLLM rollout backend.

Upstream project: https://github.com/verl-project/verl

The group reward hooks use `requires_grouped_rollouts` to defer scoring until a
complete query group is available. Validation padding is removed before grouped
scoring. The tensorboard adapter logs training metrics directly.

License: Apache-2.0. See `LICENSE` and `Notice.txt`.
