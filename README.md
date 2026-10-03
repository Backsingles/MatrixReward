# MatrixReward

MatrixReward derives training rewards from query-specific rubrics. Each response pair is judged once per rubric in a randomly chosen display order. It builds a rubric-wise comparison matrix, combines matrix-derived weights with human importance weights, and converts spatial distances into rewards for GRPO training. The training framework is included in `third_party/verl`.

## Quick Start

Use Python 3.11 or 3.12 in a Linux CUDA environment. Run the following commands from the repository root.

**1. Install dependencies**

```bash
bash setup_verl.sh
```

**2. Prepare your input**

Each JSONL record should contain `query`, a matching single-user-message `prompt`, and `rubrics_json`: a JSON string containing 3–5 objects with `rubric` text and integer `points` from 1 to 3. Rubric-generation and judge prompts are in `prompts/`.

```bash
python prepare_verl_data.py --input /path/to/queries.jsonl --output /path/to/train.parquet
```

**3. Start training**

```bash
export TRAIN_FILE="/path/to/train.parquet"
export MODEL_PATH="Qwen/Qwen3-8B"
export JUDGE_API_URL="<chat-completions-endpoint>"
export JUDGE_API_KEY="<runtime-key>"  # Optional if authentication is not required
export NUM_GPUS=8
bash run_training.sh
```

Training settings are in [`configs/verl.yaml`](configs/verl.yaml). Use `bash run_training.sh --check` to inspect the resolved configuration.
