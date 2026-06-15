# POC 0 â€” KV Visibility Map Report

Status: **MECHANISM PASS / ABSOLUTE QUALITY FAIL**

## Validated
The experiment validates the existence of block-level visibility in long-context prompts. The visibility policy achieved **100% gold block recall** while reducing the active context by **73%** and estimated KV reads by **68%**.

The reduced visibility replay outperformed the full-context baseline by **4x** on `Qwen2.5-0.5B-Instruct` (39.6% vs 10.2% Exact Match), demonstrating that context culling significantly reduces attention dispersion and distraction noise for small models.

## Not Yet Validated
The current method is an oracle-style two-pass pipeline. It requires a full-context attention pass before culling, so it does not yet reduce real serving cost in production.

The measured TTFT reduction applies to the replay pass, not to the full two-pass pipeline.

## Next Step
Build a pre-prefill visibility predictor (POC 0.1) that approximates the attention-derived hot block selection without running the full-context model first.

---

## Dataset Configuration
- **Total Questions**: 500
- **Average Blocks per Prompt**: 50.0
- **Block Size**: 128 tokens

## Quality Metrics

| Metric | Full Context | KV Visibility | Random Baseline | BM25 Baseline | Target | Status |
|---|---|---|---|---|---|---|
| **Exact Match** | 10.2% | **39.6%** | 18.0% | 30.8% | &ge; 95.0% | **FAIL** |
| **Numeric Preservation** | 70.0% | **76.2%** | - | - | &ge; 99.0% | **FAIL** |
| **Contradiction Accuracy** | - | **33.0%** | - | - | &ge; 95.0% | **FAIL** |

## Evidence and Alignment

- **Gold Block Recall**: 100.0% (Target: &ge; 99.0%) &rarr; **PASS**
- **Random Baseline Gap**: 21.6 pts (Target: &ge; 20.0 pts) &rarr; **PASS**
- **BM25 Comparison**: KV Visibility is +8.8 pts relative to BM25 (Target: &ge; 0.0 pts) &rarr; **PASS**

## Efficiency & KV Performance

- **Average Context Blocks**: 50.0
- **Average Kept Blocks**: 16.0
- **Estimated KV Read Reduction**: 68.0% (Target: &ge; 60.0%) &rarr; **PASS**
- **Token Reduction**: 73.0% (Target: &ge; 60.0%) &rarr; **PASS**

## Latency Metrics (TTFT)
- **Full Context TTFT**: 4984.4 ms
- **Replay Context TTFT**: 2095.6 ms
- **TTFT Reduction**: 58.0%

