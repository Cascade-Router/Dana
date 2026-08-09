---
name: Bug report
about: Report a crash, incorrect behavior, or environment failure
title: "[bug] "
labels: bug
assignees: ""
---

## Environment

| Field | Your value |
|-------|------------|
| **OS** | e.g. Windows 11 24H2 |
| **GPU / VRAM** | e.g. NVIDIA RTX 4070 12GB (or CPU-only) |
| **PyTorch version** | e.g. `2.13.0+cu126` (`python -c "import torch; print(torch.__version__)"`) |
| **CUDA / driver** | e.g. CUDA 12.6 / driver 560.x (or N/A) |
| **Python version** | e.g. 3.11.9 |
| **Dana commit / tag** | e.g. `main` @ `abc1234` |

## Summary

A clear, one-sentence description of what went wrong.

## Steps to reproduce

1.
2.
3.

## Expected behavior

What should have happened.

## Actual behavior

What happened instead.

## Error logs

Paste relevant excerpts from:

- terminal / `run.py` stderr
- `logs/dana_runtime.log` (if present)
- Live Trace / `logs/dana_telemetry.jsonl` (if relevant)

```text
(paste logs here — redact secrets, API keys, and personal paths if needed)
```

## Additional context

Screenshots, HITL ticket IDs, or related PRs/issues.
