#!/usr/bin/env bash
source /workspace/vd-venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf-cache
exec vllm serve google/gemma-3-27b-it \
  --served-model-name gemma-3-27b-it \
  --dtype bfloat16 --port 8000 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192
