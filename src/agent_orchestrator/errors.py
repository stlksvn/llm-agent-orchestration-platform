"""Explicit exception hierarchy for predictable failure handling."""


class OrchestrationError(Exception):
    """Base exception for orchestration failures."""


class ProviderError(OrchestrationError):
    """An LLM provider returned an error or an invalid response."""


class ProviderTimeoutError(ProviderError):
    """An LLM request exceeded its configured timeout."""


class StrategyError(OrchestrationError):
    """A strategy could not complete its workflow."""
