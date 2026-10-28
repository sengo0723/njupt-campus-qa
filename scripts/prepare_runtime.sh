#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models results
.venv/bin/python scripts/prepare_models.py
.venv/bin/python -m mlx_lm convert --hf-path models/qwen3-base-hf --mlx-path models/qwen3-base-mlx --dtype bfloat16
.venv/bin/python scripts/vendor/llama.cpp/convert_hf_to_gguf.py models/qwen3-base-mlx --outfile models/qwen3-base-q8_0.gguf --outtype q8_0
echo '基础模型资源已准备；运行 scripts/train_and_deploy.sh 训练并注册模型，然后启动网页。'
