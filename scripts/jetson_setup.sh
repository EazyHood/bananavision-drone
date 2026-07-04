#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[api,opencv,geo]"

echo "For training on the Jetson, also install: pip install -e '.[ml]'"
echo "For TensorRT export, use the CUDA/TensorRT stack supplied for your JetPack version."
