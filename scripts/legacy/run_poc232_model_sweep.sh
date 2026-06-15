#!/bin/bash
set -u
set -o pipefail

#############################################
# POC 2.3.2 - MODEL SWEEP FOR G/H CATEGORIES
# 4 models / G/H-only workload / 116 queries
#############################################

RUN_NAME="poc_2_3_2_model_sweep"
RUN_DIR="artifacts/runs/${RUN_NAME}"
WORKLOAD_FILE="data/workloads/gh_only_116.jsonl"
CORPUS_DIR="data/corpus_poc2/index"
PYTHON="/home/taurus_silver/miniconda3/envs/poseidon_wsl/bin/python"

mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/sweep.log"

echo "==================================================" | tee "${LOG_FILE}"
echo "POC 2.3.2 Model Sweep Run" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "Run dir: ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

# 1. BUILD WORKLOAD
echo "" | tee -a "${LOG_FILE}"
echo "[1/3] Building G/H-only workload..." | tee -a "${LOG_FILE}"
"${PYTHON}" -u build_poc232_gh_workload.py \
  --corpus "${CORPUS_DIR}" \
  --output "${WORKLOAD_FILE}" \
  2>&1 | tee -a "${LOG_FILE}"

# List of models to evaluate
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
)

MODEL_IDS=(
  "qwen_1_5b"
)

# Function to clean up any running vLLM servers
cleanup_vllm() {
  echo "Cleaning up any existing vLLM processes..." | tee -a "${LOG_FILE}"
  pkill -9 -f vllm 2>/dev/null || true
  pkill -9 -f api_server 2>/dev/null || true
  pkill -9 -f run_pccc_benchmark 2>/dev/null || true
  pkill -9 -f resource_tracker 2>/dev/null || true
  # Force kill anything on port 8000 to avoid port binding conflicts
  fuser -k -9 8000/tcp 2>/dev/null || true
  sleep 5
}

# 2. ITERATE OVER MODELS
echo "" | tee -a "${LOG_FILE}"
echo "[2/3] Starting model sweep..." | tee -a "${LOG_FILE}"

for i in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$i]}"
  MODEL_ID="${MODEL_IDS[$i]}"
  
  echo "--------------------------------------------------" | tee -a "${LOG_FILE}"
  echo "Processing model: ${MODEL_NAME} (${MODEL_ID})" | tee -a "${LOG_FILE}"
  echo "--------------------------------------------------" | tee -a "${LOG_FILE}"
  
  cleanup_vllm
  
  # Clear cache to prevent cross-model cache hits
  echo "Clearing caches..." | tee -a "${LOG_FILE}"
  rm -f data/corpus_poc2/cache/*.json
  
  # Start vLLM server with context length limits matching the scheduler limit to save VRAM/KV Cache
  if [[ "${MODEL_ID}" == "qwen_1_5b" ]]; then
    MAX_LEN="4096"
    MEM_UTIL="0.80"
  elif [[ "${MODEL_ID}" == "qwen_3b" ]]; then
    MAX_LEN="4096"
    MEM_UTIL="0.80"
  elif [[ "${MODEL_ID}" == "qwen_7b_gptq" ]]; then
    MAX_LEN="2048"
    MEM_UTIL="0.80"
  else
    MAX_LEN="16384"
    MEM_UTIL="0.80"
  fi
  
  echo "Starting vLLM server for ${MODEL_NAME}..." | tee -a "${LOG_FILE}"
  bash experiments/kv_visibility_poc1/start_vllm.sh "${MODEL_NAME}" 8000 "${MAX_LEN}" "${MEM_UTIL}" float16 > "${RUN_DIR}/vllm_server_${MODEL_ID}.log" 2>&1 &
  VLLM_PID=$!
  
  # Wait for vLLM to start
  echo "Waiting for server to respond..." | tee -a "${LOG_FILE}"
  SERVER_READY=0
  for attempt in {1..150}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"model": "'"${MODEL_NAME}"'", "prompt": "Hello", "max_tokens": 1}' http://localhost:8000/v1/completions || echo "000")
    if [ "${STATUS}" -eq 200 ]; then
      echo "Server is ready and fully warmed up!" | tee -a "${LOG_FILE}"
      SERVER_READY=1
      break
    fi
    sleep 3
  done
  
  if [ ${SERVER_READY} -eq 0 ]; then
    echo "ERROR: Server failed to start for ${MODEL_NAME}." | tee -a "${LOG_FILE}"
    continue
  fi
  
  # Record peak VRAM after load
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '\r' > "${RUN_DIR}/vram_${MODEL_ID}.txt"
    echo "VRAM used after loading: $(cat ${RUN_DIR}/vram_${MODEL_ID}.txt) MB" | tee -a "${LOG_FILE}"
  else
    echo "N/A" > "${RUN_DIR}/vram_${MODEL_ID}.txt"
  fi
  
  # Run the benchmark
  RESULTS_FILE="${RUN_DIR}/results_${MODEL_ID}.jsonl"
  SUMMARY_FILE="${RUN_DIR}/summary_${MODEL_ID}.md"
  rm -f "${RESULTS_FILE}"
  
  echo "Running benchmark..." | tee -a "${LOG_FILE}"
  set +e
  "${PYTHON}" -u run_pccc_benchmark.py \
    --run-name "${RUN_NAME}" \
    --corpus "${CORPUS_DIR}" \
    --workload "${WORKLOAD_FILE}" \
    --model-name "${MODEL_NAME}" \
    --modes pccc_cache_scheduler_prefix \
    --resume false \
    --output "${RESULTS_FILE}" \
    --summary "${SUMMARY_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  BENCH_EXIT=$?
  set -e
  echo "Benchmark exited with code: ${BENCH_EXIT}" | tee -a "${LOG_FILE}"
  
  # Stop server
  cleanup_vllm
done

# 3. EVALUATE RESULTS
echo "" | tee -a "${LOG_FILE}"
echo "[3/3] Generating sweep evaluation report..." | tee -a "${LOG_FILE}"

"${PYTHON}" -u eval_poc232_model_sweep.py \
  --results-dir "${RUN_DIR}" \
  --output-report "${RUN_DIR}/sweep_report.md" \
  2>&1 | tee -a "${LOG_FILE}"

echo "==================================================" | tee -a "${LOG_FILE}"
echo "POC 2.3.2 Model Sweep completed at: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

