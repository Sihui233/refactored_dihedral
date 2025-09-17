#!/usr/bin/env bash
set -euo pipefail
# Optional: ensure you’re at project root so Python can find the package
# cd "$(dirname "$0")"

ARGS=(
  0.001        # learning_rate
  0.0000001     # weight_decay
  18           # p
  18           # batch_size
  adam         # optimizer
  1000         # epochs
  57           # k
  random_random # batch_experiment
  128          # d_model
  32           # d_head
  4            # num_heads
  2            # n_ctx
  ReLU         # act_type
  1.0          # attn_coeff
  8            # nn_multiplier
  2            # num_mlp_layers
  0            # seed1 (add more seeds here if you want)
)


python3 -m run.run_training_Transformers "${ARGS[@]}"
