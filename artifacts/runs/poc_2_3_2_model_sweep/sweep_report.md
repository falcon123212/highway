# POC 2.3.2 Model Sweep Report
## G/H Categories LLM Synthesis Evaluation

| ModÃ¨le | EM Cat G | EM Cat H | EM Global (G/H) | Latence p50 (ms) | VRAM Peak (MB) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen 2.5 1.5B | 80.00% | 66.67% | **75.00%** | 1797.9 | 7417 MB | **FAIL** |
| Qwen 2.5 3B | 0.00% | 0.00% | **0.00%** | 43.2 | 7790 MB | **FAIL** |
| Qwen 2.5 7B GPTQ-Int4 | 0.00% | 0.00% | **0.00%** | 34.1 | 7775 MB | **FAIL** |
| Mistral 7B GPTQ-Int4 | 0.00% | 0.00% | **0.00%** | 40.1 | 7790 MB | **FAIL** |

### Conclusion & Observations
- **Target Gate**: LLM-required EM (G/H) $\ge 85\%$, Overall EM $\ge 90\%$ (for G/H).
- Look at VRAM vs accuracy tradeoffs to select the best production model.
