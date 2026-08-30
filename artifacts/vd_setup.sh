#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python -m venv /workspace/vd-venv
source /workspace/vd-venv/bin/activate
pip install -q -U pip
pip install -q vllm
pip install -q pandas numpy tqdm matplotlib einops scipy openai ipykernel
python - << "PY"
import torch
assert torch.cuda.is_available(), "CUDA unavailable in venv"
print(f"venv OK: torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY
echo VD_SETUP_DONE
