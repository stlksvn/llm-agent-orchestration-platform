# Changelog

## 0.2.0

- Added a shared bounded retry policy with exponential backoff for transient provider failures and timeouts.
- Added JSON routing-decision parsing and allow-list validation for centralized and distributed strategies.
- Added deterministic fallback routing when model output is malformed or proposes an unavailable worker.
- Added retry-attempt metrics to agent results and run events.
- Expanded tests for retries, malformed routing output, and strategy-specific routing semantics.

## 0.1.0

- Initial provider-neutral orchestration runtime with Ollama, OpenAI-compatible, and mock providers.
- Added sequential, parallel, centralized, and distributed strategy prototypes.
