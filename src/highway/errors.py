from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class HighwayError(Exception):
    """Base class for structured Highway domain errors."""

    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    code: ClassVar[str] = "HIGHWAY_ERROR"
    legacy_prefix: ClassVar[str] = "EXECUTION_ERROR"

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message or self.code)

    def to_legacy_answer(self) -> str:
        return f"{self.legacy_prefix}:{self.code}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message or self.code,
            "details": dict(self.details),
            "legacy_answer": self.to_legacy_answer(),
        }

    @classmethod
    def from_code(cls, code: str, message: str = "", details: dict[str, Any] | None = None) -> "HighwayError":
        error_cls = _ERRORS_BY_CODE.get(code, HighwayError)
        return error_cls(message=message, details=details or {})


class RetrievalError(HighwayError):
    code = "RETRIEVAL_ERROR"


class CompilationError(HighwayError):
    code = "COMPILATION_ERROR"


class ContextOverflowError(CompilationError):
    code = "CONTEXT_OVERFLOW"


class LLMUnavailableError(HighwayError):
    code = "LLM_UNAVAILABLE"


class MalformedJSONError(HighwayError):
    code = "MALFORMED_JSON"


class ValidationError(HighwayError):
    code = "VALIDATION_ERROR"


class ConfigurationError(HighwayError):
    code = "CONFIGURATION_ERROR"


_ERRORS_BY_CODE: dict[str, type[HighwayError]] = {
    error_cls.code: error_cls
    for error_cls in (
        RetrievalError,
        CompilationError,
        ContextOverflowError,
        LLMUnavailableError,
        MalformedJSONError,
        ValidationError,
        ConfigurationError,
    )
}
