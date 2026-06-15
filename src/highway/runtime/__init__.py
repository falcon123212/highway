"""Runtime orchestration components.

The package facade is intentionally lazy: storage modules import low-level runtime
helpers such as ``HardwareBudget``, and eager imports here would create cycles
through ``ContextEngine -> OutOfCoreIndex``.
"""

__all__ = [
    "ContextAdapter",
    "AnswerAudit",
    "AnswerContract",
    "AnswerContractCompiler",
    "AnswerVerifier",
    "ContextBlock",
    "ContextPack",
    "ContextRequest",
    "DeterministicReflectiveClient",
    "HighwayContextEngine",
    "HighwayLLMRuntime",
    "ModelProfile",
    "OllamaLLMClient",
    "SessionState",
    "TokenEconomics",
]


def __getattr__(name):
    if name in {"AnswerAudit", "AnswerContract", "AnswerContractCompiler", "AnswerVerifier"}:
        from highway.runtime.answer_contract import (
            AnswerAudit,
            AnswerContract,
            AnswerContractCompiler,
            AnswerVerifier,
        )

        return {
            "AnswerAudit": AnswerAudit,
            "AnswerContract": AnswerContract,
            "AnswerContractCompiler": AnswerContractCompiler,
            "AnswerVerifier": AnswerVerifier,
        }[name]
    if name in {"ContextAdapter", "SessionState"}:
        from highway.runtime.context_adapter import ContextAdapter, SessionState

        return {"ContextAdapter": ContextAdapter, "SessionState": SessionState}[name]
    if name in {"ContextBlock", "ContextPack", "ContextRequest", "HighwayContextEngine"}:
        from highway.runtime.context_engine import ContextBlock, ContextPack, ContextRequest, HighwayContextEngine

        return {
            "ContextBlock": ContextBlock,
            "ContextPack": ContextPack,
            "ContextRequest": ContextRequest,
            "HighwayContextEngine": HighwayContextEngine,
        }[name]
    if name in {"DeterministicReflectiveClient", "HighwayLLMRuntime"}:
        from highway.runtime.llm_runtime import DeterministicReflectiveClient, HighwayLLMRuntime

        return {
            "DeterministicReflectiveClient": DeterministicReflectiveClient,
            "HighwayLLMRuntime": HighwayLLMRuntime,
        }[name]
    if name in {"OllamaLLMClient"}:
        from highway.runtime.ollama_client import OllamaLLMClient

        return OllamaLLMClient
    if name in {"ModelProfile", "TokenEconomics"}:
        from highway.runtime.token_economics import ModelProfile, TokenEconomics

        return {"ModelProfile": ModelProfile, "TokenEconomics": TokenEconomics}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


