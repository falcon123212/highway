#!/bin/bash
# vLLM startup script for WSL2 - sets CUDA_HOME and PATH to conda-installed nvcc
# Use a clean PATH to avoid Windows paths with parentheses breaking bash parsing
CONDA_ENV=/home/taurus_silver/miniconda3/envs/poseidon_wsl
CU13_DIR=${CONDA_ENV}/lib/python3.11/site-packages/nvidia/cu13
NVCC_DIR=${CU13_DIR}/bin
PYTHON=${CONDA_ENV}/bin/python

export CUDA_HOME=${CU13_DIR}
export CUDA_PATH=${CU13_DIR}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_V1=0
export LD_LIBRARY_PATH=${CU13_DIR}/lib:${CONDA_ENV}/lib:${LD_LIBRARY_PATH}
export PATH=${NVCC_DIR}:${CONDA_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

exec ${PYTHON} -m vllm.entrypoints.openai.api_server \
    --model "$1" \
    --port "${2:-8000}" \
    --max-model-len "${3:-55000}" \
    --gpu-memory-utilization "${4:-0.90}" \
    --dtype "${5:-float16}" \
    --trust-remote-code \
    --attention-backend FLASH_ATTN \
    --enforce-eager

