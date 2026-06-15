#!/bin/bash
# ===========================================================================
# POC 2.3.3 â€” Compute Kernels DÃ©terministes â€” Validation Script
# ===========================================================================
# ExÃ©cute le benchmark ciblÃ© sur les 24 queries G/H avec compute kernels.
# Pas de serveur vLLM nÃ©cessaire (les kernels sont purement CPU).
#
# Gates de validation :
#   G/H EM Global       >= 95%
#   COMPUTE_COMPARISON EM >= 95%
#   COMPUTE_AGGREGATION EM >= 95%
#   False NOT_FOUND      = 0%
#   Exec Error â†’ NOT_FOUND = 0%
#   LLM Call Rate on G/H = 0%
#   Verifier Pass        >= 99%
#   Latency p95          < 50 ms
# ===========================================================================

set -euo pipefail

# Paths (WSL)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="/home/taurus_silver/miniconda3/envs/poseidon_wsl/bin/python"
CORPUS="${PROJECT_DIR}/data/corpus_poc2/index"
WORKLOAD="${PROJECT_DIR}/data/workloads/gh_only_116.jsonl"
RUN_NAME="poc_2_3_3_kernels"
RUN_DIR="${PROJECT_DIR}/artifacts/runs/${RUN_NAME}"
OUTPUT="${RUN_DIR}/results.jsonl"
SUMMARY="${RUN_DIR}/summary.md"
EVAL_OUTPUT="${RUN_DIR}/report.md"
EVAL_JSON="${RUN_DIR}/metrics.json"

echo "=========================================="
echo "POC 2.3.3 â€” Compute Kernels Benchmark"
echo "=========================================="

# Create run directory
mkdir -p "${RUN_DIR}"

# Clear previous results if any
if [ -f "${OUTPUT}" ]; then
    echo "Clearing previous results..."
    rm -f "${OUTPUT}"
fi

echo "Running benchmark on 24 G/H queries..."
echo "  Corpus:   ${CORPUS}"
echo "  Workload: ${WORKLOAD}"
echo "  Output:   ${OUTPUT}"

# Run benchmark (single mode: pccc_cache_scheduler_prefix)
# No vLLM needed â€” kernels are pure Python, but the scheduler init 
# still tries to connect to vLLM for potential LLM fallback.
# We pass --model-name but won't use it for G/H queries.
cd "${PROJECT_DIR}"
${PYTHON} run_pccc_benchmark.py \
    --run-name "${RUN_NAME}" \
    --corpus "${CORPUS}" \
    --workload "${WORKLOAD}" \
    --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
    --modes "pccc_cache_scheduler_prefix" \
    --max-active-tokens 1200 \
    --enable-answer-cache false \
    --enable-proof-cache false \
    --enable-evidence-cache false \
    --enable-compiled-prompt-cache false \
    --resume false \
    --output "${OUTPUT}" \
    --summary "${SUMMARY}"

echo ""
echo "Benchmark completed. Running evaluation..."

# Run evaluation
${PYTHON} eval_pccc_benchmark.py \
    --input "${OUTPUT}" \
    --output "${EVAL_OUTPUT}" \
    --export-json "${EVAL_JSON}"

echo ""
echo "=========================================="
echo "Results saved to:"
echo "  Raw:    ${OUTPUT}"
echo "  Report: ${EVAL_OUTPUT}"
echo "  JSON:   ${EVAL_JSON}"
echo "=========================================="

# Quick gate check from JSON
echo ""
echo "=== GATE CHECK ==="
${PYTHON} -c "
import json
with open('${EVAL_JSON}') as f:
    data = json.load(f)
pccc = data['modes'].get('pccc_cache_scheduler_prefix', {})
gates = [
    ('G/H EM Global',           pccc.get('gh_em_global', 0),           '>= 95%',  pccc.get('gh_em_global', 0) >= 95),
    ('COMPUTE_COMPARISON EM',   pccc.get('compute_comparison_em', 0),  '>= 95%',  pccc.get('compute_comparison_em', 0) >= 95),
    ('COMPUTE_AGGREGATION EM',  pccc.get('compute_aggregation_em', 0), '>= 95%',  pccc.get('compute_aggregation_em', 0) >= 95),
    ('False NOT_FOUND',         pccc.get('false_not_found_rate', 0),   '= 0%',    pccc.get('false_not_found_rate', 0) == 0),
    ('Exec Error Conv.',        pccc.get('exec_error_conversion_rate',0),'= 0%',  pccc.get('exec_error_conversion_rate', 0) == 0),
    ('LLM Call Rate on G/H',    pccc.get('gh_llm_call_rate', 0),       '= 0%',    pccc.get('gh_llm_call_rate', 0) == 0),
    ('Verifier Pass',           pccc.get('verifier_pass_rate', 0),     '>= 99%',  pccc.get('verifier_pass_rate', 0) >= 99),
    ('Latency p95',             pccc.get('p95_latency_ms', 0),         '< 50ms',  pccc.get('p95_latency_ms', 0) < 50),
]
all_pass = True
for name, val, target, passed in gates:
    status = 'PASS' if passed else 'FAIL'
    if not passed:
        all_pass = False
    print(f'  {status} | {name}: {val:.2f} (target {target})')
print()
if all_pass:
    print('ALL GATES PASSED âœ“')
else:
    print('SOME GATES FAILED âœ—')
"

echo "Done."

