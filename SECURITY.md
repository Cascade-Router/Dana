# Security Policy

## Local privacy guarantees

Dana is designed as a **local-first** agentic voice OS:

- Cognition on the voice critical path is intended to stay on-device (e.g. Ollama), not round-trip through a required cloud LLM.
- Machine-local artifacts (`vault/`, `execution_jail/`, `logs/`, `.env`, `settings.json`, model weights) are gitignored and must not be committed.
- System writes and patch-ledger updates go through HITL ticket gates and **fail closed** when unapproved — unapproved / denied tickets must not execute.

Cloud or optional bridges (research swarm, Hugging Face Space demos, GitHub escalation) are separate surfaces; treat API keys and tokens as secrets and never commit them.

## Supported versions

Security fixes are applied on the active `main` branch of this repository. If you are running a fork or an older commit, rebase or cherry-pick before filing a private report when possible.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately so we can assess and patch before disclosure:

1. Prefer GitHub **[Private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)** on this repository (Security → Report a vulnerability), when enabled.
2. Otherwise email the maintainers via the contact listed on the GitHub org/profile for **Cascade-Router / AMIXXM**, with subject line `[Dana security]`.

Include:

- Affected commit / tag
- Impact (confidentiality, integrity, availability; jail escape; HITL bypass; secret leakage)
- Minimal reproduction steps
- Whether a fix or workaround is already known

We will acknowledge actionable reports and coordinate a fix + disclosure timeline. Please give us a reasonable window before public discussion.
