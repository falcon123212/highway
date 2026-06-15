#!/bin/bash
set -u
set -o pipefail

#############################################
# POC 2.3 NIGHT SAFE RUN
# 500 queries / 3 modes / timeout 6h45
#############################################

RUN_NAME="poc_2_3_night_safe_mixed_execution"
RUN_DIR="artifacts/runs/${RUN_NAME}"
WORKLOAD_DIR="data/workloads"
WORKLOAD_FILE="${WORKLOAD_DIR}/mixed_runtime_500_safe.jsonl"
CORPUS_DIR="data/corpus_poc2/index"

TIMEOUT_LIMIT="405m"
VLLM_HOST="localhost"
MODEL_ID="qwen_0_5b"
SEED=42
PYTHON="/home/taurus_silver/miniconda3/envs/poseidon_wsl/bin/python"

mkdir -p "${RUN_DIR}"
mkdir -p "${WORKLOAD_DIR}"

LOG_FILE="${RUN_DIR}/night.log"
RESULTS_FILE="${RUN_DIR}/results.jsonl"
SUMMARY_FILE="${RUN_DIR}/summary.md"
FINAL_REPORT="${RUN_DIR}/final_report.md"
FINAL_METRICS="${RUN_DIR}/final_metrics.json"
STATUS_FILE="${RUN_DIR}/RUN_STATUS.txt"

echo "==================================================" | tee "${LOG_FILE}"
echo "POC 2.3 Night Safe Run" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "Run dir: ${RUN_DIR}" | tee -a "${LOG_FILE}"
echo "Timeout: ${TIMEOUT_LIMIT}" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

#############################################
# 0. PREFLIGHT CHECKS
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[0/6] Preflight checks..." | tee -a "${LOG_FILE}"

if [ ! -d "${CORPUS_DIR}" ]; then
  echo "ERROR: Corpus directory not found: ${CORPUS_DIR}" | tee -a "${LOG_FILE}"
  echo "FAILED_PRECHECK_CORPUS_MISSING" > "${STATUS_FILE}"
  exit 1
fi

if [ ! -f "build_poc23_workload.py" ]; then
  echo "ERROR: Missing build_poc23_workload.py" | tee -a "${LOG_FILE}"
  echo "FAILED_PRECHECK_BUILD_SCRIPT_MISSING" > "${STATUS_FILE}"
  exit 1
fi

if [ ! -f "run_pccc_benchmark.py" ]; then
  echo "ERROR: Missing run_pccc_benchmark.py" | tee -a "${LOG_FILE}"
  echo "FAILED_PRECHECK_RUNNER_MISSING" > "${STATUS_FILE}"
  exit 1
fi

if [ ! -f "eval_pccc_benchmark.py" ]; then
  echo "ERROR: Missing eval_pccc_benchmark.py" | tee -a "${LOG_FILE}"
  echo "FAILED_PRECHECK_EVAL_MISSING" > "${STATUS_FILE}"
  exit 1
fi

echo "Corpus and scripts found." | tee -a "${LOG_FILE}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU check:" | tee -a "${LOG_FILE}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | tee -a "${LOG_FILE}"
else
  echo "WARNING: nvidia-smi not found. Continuing anyway." | tee -a "${LOG_FILE}"
fi

echo "Starting vLLM server..." | tee -a "${LOG_FILE}"
bash experiments/kv_visibility_poc1/start_vllm.sh Qwen/Qwen2.5-0.5B-Instruct 8000 32768 0.50 float16 > "${RUN_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

# Register trap to kill vLLM on exit
trap 'echo "Stopping vLLM server..."; kill ${VLLM_PID} 2>/dev/null; pkill -f vllm.entrypoints.openai.api_server; exit' EXIT

echo "Waiting for vLLM server to start (polling http://${VLLM_HOST}:8000/v1/models)..." | tee -a "${LOG_FILE}"
for i in {1..150}; do
  if curl -s "http://${VLLM_HOST}:8000/v1/models" >/dev/null; then
    echo "vLLM server is ready and responding!" | tee -a "${LOG_FILE}"
    break
  fi
  sleep 2
done

if ! curl -s "http://${VLLM_HOST}:8000/v1/models" >/dev/null; then
  echo "ERROR: vLLM server failed to start. Check logs at: ${RUN_DIR}/vllm_server.log" | tee -a "${LOG_FILE}"
  echo "FAILED_VLLM_START" > "${STATUS_FILE}"
  exit 1
fi

#############################################
# 1. BUILD WORKLOAD
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[1/6] Building 500-query mixed workload..." | tee -a "${LOG_FILE}"

"${PYTHON}" -u build_poc23_workload.py \
  --corpus "${CORPUS_DIR}" \
  --output "${WORKLOAD_FILE}" \
  --n-extractive 125 \
  --n-not-found 75 \
  --n-suffix-conflict 75 \
  --n-llm-synthesis 150 \
  --n-long-context 50 \
  --n-cache-replay 25 \
  --seed "${SEED}" \
  2>&1 | tee -a "${LOG_FILE}"

BUILD_EXIT=${PIPESTATUS[0]}

if [ ${BUILD_EXIT} -ne 0 ]; then
  echo "ERROR: Workload generation failed." | tee -a "${LOG_FILE}"
  echo "FAILED_WORKLOAD_BUILD" > "${STATUS_FILE}"
  exit 1
fi

if [ ! -f "${WORKLOAD_FILE}" ]; then
  echo "ERROR: Workload file not created: ${WORKLOAD_FILE}" | tee -a "${LOG_FILE}"
  echo "FAILED_WORKLOAD_FILE_MISSING" > "${STATUS_FILE}"
  exit 1
fi

WORKLOAD_COUNT=$(wc -l < "${WORKLOAD_FILE}")
echo "Workload created: ${WORKLOAD_FILE}" | tee -a "${LOG_FILE}"
echo "Workload line count: ${WORKLOAD_COUNT}" | tee -a "${LOG_FILE}"

#############################################
# 2. RUN BENCHMARK WITH HARD TIMEOUT
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[2/6] Launching benchmark with timeout ${TIMEOUT_LIMIT}..." | tee -a "${LOG_FILE}"

set +e

timeout "${TIMEOUT_LIMIT}" "${PYTHON}" -u run_pccc_benchmark.py \
  --run-name "${RUN_NAME}" \
  --corpus "${CORPUS_DIR}" \
  --workload "${WORKLOAD_FILE}" \
  --models "${MODEL_ID}" \
  --engine vllm \
  --vllm-host "${VLLM_HOST}" \
  --modes pccc_no_cache,pccc_cache_scheduler_prefix,hybrid_topk_llm_baseline_stratified_150 \
  --max-active-tokens 1200 \
  --search-top-k 50 \
  --fallback-search-top-k 150 \
  --max-kept 6 \
  --fallback-max-kept 12 \
  --enable-answer-cache true \
  --enable-proof-cache true \
  --enable-evidence-cache true \
  --enable-compiled-prompt-cache true \
  --enable-prefix-friendly-compiler true \
  --enable-long-context-fallback true \
  --enable-output-verifier true \
  --temperature 0 \
  --max-new-tokens 96 \
  --save-every 10 \
  --resume true \
  --max-retries 1 \
  --timeout 45 \
  --seed "${SEED}" \
  --output "${RESULTS_FILE}" \
  --summary "${SUMMARY_FILE}" \
  2>&1 | tee -a "${LOG_FILE}"

BENCH_EXIT=${PIPESTATUS[0]}
set -e

echo "" | tee -a "${LOG_FILE}"
echo "Benchmark exit code: ${BENCH_EXIT}" | tee -a "${LOG_FILE}"

if [ ${BENCH_EXIT} -eq 124 ]; then
  echo "WARNING: Benchmark stopped by hard timeout ${TIMEOUT_LIMIT}." | tee -a "${LOG_FILE}"
  echo "TIMEOUT_PARTIAL_RESULTS" > "${STATUS_FILE}"
elif [ ${BENCH_EXIT} -ne 0 ]; then
  echo "WARNING: Benchmark exited with non-zero status: ${BENCH_EXIT}" | tee -a "${LOG_FILE}"
  echo "BENCHMARK_NONZERO_EXIT_${BENCH_EXIT}" > "${STATUS_FILE}"
else
  echo "Benchmark completed successfully." | tee -a "${LOG_FILE}"
  echo "BENCHMARK_COMPLETED" > "${STATUS_FILE}"
fi

#############################################
# 3. CHECK RESULTS FILE
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[3/6] Checking results..." | tee -a "${LOG_FILE}"

if [ ! -f "${RESULTS_FILE}" ]; then
  echo "ERROR: Results file missing: ${RESULTS_FILE}" | tee -a "${LOG_FILE}"
  echo "FAILED_NO_RESULTS_FILE" > "${STATUS_FILE}"
  exit 1
fi

RESULT_LINES=$(wc -l < "${RESULTS_FILE}")
echo "Results lines: ${RESULT_LINES}" | tee -a "${LOG_FILE}"

if [ "${RESULT_LINES}" -eq 0 ]; then
  echo "ERROR: Results file is empty." | tee -a "${LOG_FILE}"
  echo "FAILED_EMPTY_RESULTS" > "${STATUS_FILE}"
  exit 1
fi

#############################################
# 4. EVALUATE RESULTS
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[4/6] Running evaluation..." | tee -a "${LOG_FILE}"

set +e
"${PYTHON}" -u eval_pccc_benchmark.py \
  --input "${RESULTS_FILE}" \
  --output "${FINAL_REPORT}" \
  --export-json "${FINAL_METRICS}" \
  2>&1 | tee -a "${LOG_FILE}"

EVAL_EXIT=${PIPESTATUS[0]}
set -e

echo "Eval exit code: ${EVAL_EXIT}" | tee -a "${LOG_FILE}"

if [ ${EVAL_EXIT} -ne 0 ]; then
  echo "WARNING: Evaluation failed. Raw results are still available." | tee -a "${LOG_FILE}"
  echo "EVAL_FAILED_RESULTS_AVAILABLE" >> "${STATUS_FILE}"
else
  echo "Evaluation completed." | tee -a "${LOG_FILE}"
  echo "EVAL_COMPLETED" >> "${STATUS_FILE}"
fi

#############################################
# 5. BUILD QUICK HUMAN SUMMARY
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[5/6] Building quick summary..." | tee -a "${LOG_FILE}"

{
  echo "# POC 2.3 Night Safe â€” Quick Summary"
  echo ""
  echo "Run name: ${RUN_NAME}"
  echo "Started: see night.log"
  echo "Finished: $(date)"
  echo "Benchmark exit code: ${BENCH_EXIT}"
  echo "Evaluation exit code: ${EVAL_EXIT}"
  echo "Workload file: ${WORKLOAD_FILE}"
  echo "Workload count: ${WORKLOAD_COUNT}"
  echo "Results lines: ${RESULT_LINES}"
  echo ""
  echo "## Status"
  cat "${STATUS_FILE}" || true
  echo ""
  echo "## Files"
  echo "- Results: ${RESULTS_FILE}"
  echo "- Summary: ${SUMMARY_FILE}"
  echo "- Final report: ${FINAL_REPORT}"
  echo "- Final metrics: ${FINAL_METRICS}"
  echo "- Log: ${LOG_FILE}"
  echo ""
  echo "## What to inspect tomorrow"
  echo "- Overall EM"
  echo "- Routing Accuracy"
  echo "- LLM-required EM"
  echo "- False deterministic answer"
  echo "- False NOT_FOUND"
  echo "- KV tokens avoided"
  echo "- Prefix TTFT reduction"
  echo "- Long-context malformed output"
  echo "- OOM rate"
} > "${RUN_DIR}/quick_summary.md"

cat "${RUN_DIR}/quick_summary.md" | tee -a "${LOG_FILE}"

#############################################
# 6. PACKAGE ARTIFACTS
#############################################

echo "" | tee -a "${LOG_FILE}"
echo "[6/6] Packaging run artifacts..." | tee -a "${LOG_FILE}"

tar -czf "${RUN_DIR}.tar.gz" "${RUN_DIR}" 2>/dev/null
TAR_EXIT=$?

if [ ${TAR_EXIT} -eq 0 ]; then
  echo "Artifacts packaged: ${RUN_DIR}.tar.gz" | tee -a "${LOG_FILE}"
else
  echo "WARNING: Could not package artifacts." | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"
echo "POC 2.3 Night Safe Run finished at: $(date)" | tee -a "${LOG_FILE}"
echo "==================================================" | tee -a "${LOG_FILE}"

