#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models results
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OLLAMA_HOST=127.0.0.1:11435
.venv/bin/python scripts/manage.py model-server
.venv/bin/python -u scripts/train_lora.py --smoke > results/train_smoke.log 2>&1
.venv/bin/python -u scripts/train_lora.py > results/train_formal.log 2>&1
.venv/bin/python -u -m mlx_lm fuse --model models/qwen3-base-mlx --adapter-path models/lora-adapter --save-path models/qwen3-sft-hf > results/lora_fusion.log 2>&1
.venv/bin/python -u scripts/vendor/llama.cpp/convert_hf_to_gguf.py models/qwen3-sft-hf --outfile models/qwen3-sft-q8_0.gguf --outtype q8_0 > results/gguf_sft_conversion.log 2>&1
ollama_cmd="${OLLAMA_BIN:-ollama}"
"$ollama_cmd" create njupt-qwen3-base -f config/base.Modelfile > results/ollama_import_base.log 2>&1
"$ollama_cmd" create njupt-qwen3-sft -f config/sft.Modelfile > results/ollama_import_sft.log 2>&1
.venv/bin/python -u scripts/verify_model_artifacts.py > results/model_verification.log 2>&1
.venv/bin/python -u scripts/evaluate_sft_loss.py > results/sft_heldout_loss.log 2>&1
