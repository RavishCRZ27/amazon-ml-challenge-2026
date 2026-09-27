#!/usr/bin/env bash
# Make FlashInfer's JIT (vLLM sampler) work in .venv-llm without a system CUDA toolkit.
# 1. Pin the pip CUDA compiler toolchain to torch's CUDA (cu130): CCCL requires nvcc == headers.
# 2. Build .venv-llm/cuda: a standard-layout CUDA_HOME (bin, include, nvvm, lib64) of symlinks
#    into the pip nvidia/cu13 tree, plus the unversioned libcudart.so link the linker needs.
# src/s4l_llm.py uses .venv-llm/cuda when it exists (else falls back to vLLM's torch sampler).
set -euo pipefail
cd "$(dirname "$0")/.."
V="$PWD/.venv-llm"
CU=$("$V/bin/python" -c "import torch; print(torch.version.cuda)")          # e.g. 13.0
"$V/bin/pip" install -q "nvidia-cuda-nvcc==${CU}.*" "nvidia-cuda-crt==${CU}.*" "nvidia-nvvm==${CU}.*"
C="$V/lib/python3.12/site-packages/nvidia/cu${CU%%.*}"
H="$V/cuda"
rm -rf "$H" && mkdir -p "$H/lib64"
for d in bin include nvvm; do ln -s "$C/$d" "$H/$d"; done
for f in "$C"/lib/*; do ln -s "$f" "$H/lib64/$(basename "$f")"; done
ln -s libcudart.so.13 "$H/lib64/libcudart.so"
"$H/bin/nvcc" --version | tail -1
echo "CUDA_HOME shim ready: $H"
