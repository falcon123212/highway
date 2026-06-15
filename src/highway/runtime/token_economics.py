from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelProfile:
    name: str
    layers: int
    hidden_size: int
    bytes_per_element: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenEconomics:
    baseline_input_tokens: int
    actual_input_tokens: int
    avoided_input_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_ms: float
    total_llm_ms: float
    input_tokens_per_second: float
    output_tokens_per_second: float
    effective_tokens_per_second: float
    kv_bytes_estimated: Optional[int]
    kv_bytes_avoided_estimated: Optional[int]
    cost_estimated_usd: float
    cost_avoided_estimated_usd: float
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_measurements(
        cls,
        baseline_input_tokens: int,
        actual_input_tokens: int,
        output_tokens: int = 0,
        ttft_ms: float = 0.0,
        decode_ms: float = 0.0,
        model_profile: Optional[ModelProfile] = None,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> "TokenEconomics":
        baseline = max(0, int(baseline_input_tokens))
        actual = max(0, int(actual_input_tokens))
        output = max(0, int(output_tokens))
        avoided = max(0, baseline - actual)
        ttft = max(0.0, float(ttft_ms))
        decode = max(0.0, float(decode_ms))
        total = ttft + decode
        warnings: list[str] = []

        input_tps = actual / (ttft / 1000.0) if ttft > 0.0 else 0.0
        output_tps = output / (decode / 1000.0) if decode > 0.0 else 0.0
        effective_tps = baseline / (total / 1000.0) if total > 0.0 else 0.0

        kv_bytes = None
        kv_avoided = None
        if model_profile is None:
            warnings.append("model_profile_missing")
        else:
            per_token = (
                int(model_profile.layers)
                * int(model_profile.hidden_size)
                * 2
                * int(model_profile.bytes_per_element)
            )
            kv_bytes = actual * per_token
            kv_avoided = avoided * per_token

        input_cost = actual / 1_000_000.0 * float(input_cost_per_million)
        output_cost = output / 1_000_000.0 * float(output_cost_per_million)
        avoided_cost = avoided / 1_000_000.0 * float(input_cost_per_million)

        return cls(
            baseline_input_tokens=baseline,
            actual_input_tokens=actual,
            avoided_input_tokens=avoided,
            output_tokens=output,
            ttft_ms=ttft,
            decode_ms=decode,
            total_llm_ms=total,
            input_tokens_per_second=input_tps,
            output_tokens_per_second=output_tps,
            effective_tokens_per_second=effective_tps,
            kv_bytes_estimated=kv_bytes,
            kv_bytes_avoided_estimated=kv_avoided,
            cost_estimated_usd=round(input_cost + output_cost, 12),
            cost_avoided_estimated_usd=round(avoided_cost, 12),
            warnings=warnings,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
