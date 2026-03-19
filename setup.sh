#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel

if command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip install "bitsandbytes==0.42.0"
  python -m pip install "auto-gptq==0.7.1"
else
  echo "CUDA not detected; skipping bitsandbytes and auto-gptq preinstall."
fi

python -m pip install "torch==2.1.0"
python -m pip install -r requirements.txt
