from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable
from urllib import error, request

from highway.runtime.llm_runtime import estimate_tokens


Transport = Callable[[Dict[str, Any], float], Iterable[Dict[str, Any]]]


def _default_transport(base_url: str) -> Transport:
    endpoint = base_url.rstrip("/") + "/api/generate"

    def _transport(payload: Dict[str, Any], timeout_s: float) -> Iterable[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)

    return _transport


@dataclass(frozen=True)
class OllamaLLMClient:
    model: str = "qwen2.5:0.5b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0
    temperature: float = 0.0
    num_predict: int = 256
    transport: Transport | None = None

    @property
    def model_name(self) -> str:
        return self.model

    def answer(
        self,
        prompt: str,
        query_ir: Dict[str, Any],
        evidence: list[Dict[str, Any]],
        expected_answer: str | None = None,
        query_id: str = "ollama",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del query_ir, evidence, expected_answer, query_id
        started = time.perf_counter()
        input_tokens = estimate_tokens(prompt)
        chunks: list[str] = []
        final: Dict[str, Any] = {}
        first_token_at: float | None = None
        answer_contract = kwargs.get("answer_contract")
        max_output_tokens = kwargs.get("max_output_tokens")
        if max_output_tokens is None and answer_contract is not None:
            max_output_tokens = getattr(answer_contract, "max_output_tokens", None)
        num_predict_requested = int(max_output_tokens or self.num_predict)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict_requested,
            },
        }
        transport = self.transport or _default_transport(self.base_url)
        try:
            for event in transport(payload, self.timeout_s):
                text = str(event.get("response", ""))
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(text)
                if event.get("done"):
                    final = dict(event)
        except (OSError, error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            return self._unavailable(prompt, started, f"ollama_unavailable:{exc}", num_predict_requested)

        total_ms = (time.perf_counter() - started) * 1000.0
        raw_text = "".join(chunks)
        output_tokens = int(final.get("eval_count") or estimate_tokens(raw_text) if raw_text else 0)
        prompt_tokens = int(final.get("prompt_eval_count") or input_tokens)
        ttft_ms = (first_token_at - started) * 1000.0 if first_token_at is not None else total_ms
        eval_duration_ns = float(final.get("eval_duration") or 0.0)
        prompt_duration_ns = float(final.get("prompt_eval_duration") or 0.0)
        if prompt_duration_ns > 0.0:
            ttft_ms = prompt_duration_ns / 1_000_000.0
        if eval_duration_ns > 0.0:
            decode_ms = eval_duration_ns / 1_000_000.0
        else:
            decode_ms = max(0.0, total_ms - ttft_ms)
        total_llm_ms = max(total_ms, ttft_ms + decode_ms)
        return {
            "available": True,
            "model_name": self.model,
            "reasoning": "",
            "answer": raw_text,
            "raw_text": raw_text,
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": max(0.0, ttft_ms),
            "decode_ms": max(0.0, decode_ms),
            "total_ms": max(0.0, total_llm_ms),
            "input_tokens_per_second": prompt_tokens / (ttft_ms / 1000.0) if ttft_ms > 0.0 else 0.0,
            "output_tokens_per_second": output_tokens / (decode_ms / 1000.0) if decode_ms > 0.0 else 0.0,
            "num_predict_requested": num_predict_requested,
            "output_stop_reason": str(final.get("done_reason") or final.get("finish_reason") or "done"),
        }

    def _unavailable(
        self,
        prompt: str,
        started: float,
        reason: str,
        num_predict_requested: int | None = None,
    ) -> Dict[str, Any]:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "available": False,
            "skip_reason": reason,
            "model_name": self.model,
            "reasoning": "",
            "answer": "",
            "raw_text": "",
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": 0,
            "ttft_ms": elapsed_ms,
            "decode_ms": 0.0,
            "total_ms": elapsed_ms,
            "input_tokens_per_second": 0.0,
            "output_tokens_per_second": 0.0,
            "num_predict_requested": int(num_predict_requested or self.num_predict),
            "output_stop_reason": "unavailable",
        }
