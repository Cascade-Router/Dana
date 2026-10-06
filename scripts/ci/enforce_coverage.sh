#!/usr/bin/env bash
# Test parity gate: the full pytest suite under coverage, failing if line
# coverage of dana/ drops below the floor. Extra arguments go to pytest
# (CI passes its marker/timeout/junit flags through here).
#
# The floor is a ratchet, not a target: total coverage measured 61.0% on
# 2026-10-06 (Windows) after the legacy-code removal, up from 52.8%. Raise it
# whenever a change lifts coverage; never lower it to get a change through.
set -euo pipefail

COVERAGE_FLOOR="${COVERAGE_FLOOR:-60}"

cd "$(dirname "$0")/../.."
exec python -m pytest \
  --cov=dana \
  --cov-report=xml \
  --cov-report=term:skip-covered \
  --cov-fail-under="${COVERAGE_FLOOR}" \
  "$@"
