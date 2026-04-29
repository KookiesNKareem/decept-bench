# Security Policy

## Reporting a vulnerability

DECEPT-Bench is an evaluation benchmark, not a deployed service. The
relevant security concerns are:

1. **Contamination / score-gaming patterns** — anyone discovering a way to game the scoring formula or contaminate the public split should report privately by opening a GitHub Issue with the `security` label. We will move it to private channels and address before public disclosure.

2. **Submission-server vulnerabilities** — if the public submission server (`server/app.py`) is found to leak private test items or accept unvalidated outputs, report privately.

## Scope

In scope:
- Submission-server endpoints
- Scoring formula correctness
- Hidden-test-set isolation
- Repository content (no committed secrets)

Out of scope:
- Vulnerabilities in upstream dependencies (vLLM, anthropic, openai, etc.) — please report to the upstream maintainers
- Adversarial attacks on the benchmark items themselves; that's the benchmark's *purpose*