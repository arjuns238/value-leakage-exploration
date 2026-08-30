#!/usr/bin/env bash
# Runs ON the vast.ai pod. Installs the Jupyter + interp stack and starts
# JupyterLab bound to loopback only (you reach it through an SSH tunnel).
set -euo pipefail

: "${JUPYTER_TOKEN:?export JUPYTER_TOKEN=... before running}"
WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR"

# Keep the HF cache on the big --disk volume, not the default /root/.cache.
export HF_HOME="$WORKDIR/hf-cache"
mkdir -p "$HF_HOME"

TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"

pip install --no-cache-dir -q -U \
  jupyterlab jupyter-collaboration jupyter-mcp-tools ipykernel \
  'transformers>=5.4.0' accelerate huggingface_hub \
  transformer-lens nnsight \
  matplotlib pandas einops scikit-learn scipy

# transformer-lens / nnsight can drag in a torch that doesn't match the image's
# CUDA build. Fail loudly here rather than at 3am inside a notebook.
TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "WARNING: torch changed ${TORCH_BEFORE} -> ${TORCH_AFTER} during install." >&2
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after install - torch was clobbered"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY

cat > /etc/profile.d/hf.sh <<EOF
export HF_HOME='${HF_HOME}'
export HF_XET_HIGH_PERFORMANCE=1
EOF

cat > /usr/local/bin/start-jupyter <<EOF
#!/usr/bin/env bash
export HF_HOME='${HF_HOME}'
export HF_XET_HIGH_PERFORMANCE=1
cd '${WORKDIR}'
exec jupyter lab \\
  --ServerApp.ip=127.0.0.1 \\
  --ServerApp.port=8888 \\
  --IdentityProvider.token='${JUPYTER_TOKEN}' \\
  --ServerApp.root_dir='${WORKDIR}' \\
  --ServerApp.open_browser=False \\
  --allow-root
EOF
chmod +x /usr/local/bin/start-jupyter

# RunPod's PyTorch template starts its own JupyterLab on 8888. Evict it.
pkill -f "jupyter[- ]lab" 2>/dev/null || true
sleep 2
setsid nohup /usr/local/bin/start-jupyter > "$WORKDIR/jupyter.log" 2>&1 < /dev/null &

sleep 8
if curl -sf "http://127.0.0.1:8888/api?token=${JUPYTER_TOKEN}" > /dev/null; then
  echo "JupyterLab up on 127.0.0.1:8888  (log: $WORKDIR/jupyter.log)"
  echo "HF_HOME=$HF_HOME"
else
  echo "Jupyter did not come up. Last 40 lines:"; tail -40 "$WORKDIR/jupyter.log"; exit 1
fi
