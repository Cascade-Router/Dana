"""Offline OSWorld-style task adapter → Dana ``ReactGraphState`` + step scoring.

Converts fixture JSON (prompts, target screen states, expected action sequences)
into Dana graph inputs and maps Dana clicks / keystrokes / tool invocations to
OSWorld-style pass/fail assertions. No network / OSWorld download required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dana.schema import ReactGraphState

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Action types accepted in fixture expected_actions / Dana action_output.
_CLICK_TYPES = frozenset({"click", "double_click", "left_click", "right_click"})
_TYPE_TYPES = frozenset({"type", "type_text", "hotkey", "key_combination", "keystroke"})
_TOOL_TYPES = frozenset({"tool", "tool_call", "invoke_tool"})


def load_osworld_fixture(name: str = "minimal_osworld_task.json") -> dict[str, Any]:
    """Load an embedded OSWorld-like task fixture from ``tests/evals/fixtures/``."""
    path = _FIXTURES_DIR / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"fixture must be a JSON object: {path}")
    return raw


def _norm_action(action: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        return {}
    out = dict(action)
    at = str(out.get("type") or out.get("action") or "").strip().lower()
    if at:
        out["type"] = at
    return out


def _bbox_of(action: Mapping[str, Any]) -> list[float] | None:
    box = action.get("bbox") or action.get("box") or action.get("xyxy")
    if box is None and "x" in action and "y" in action:
        x, y = float(action["x"]), float(action["y"])
        return [x, y, x, y]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [float(v) for v in box[:4]]
    return None


def _point_in_bbox(x: float, y: float, box: Sequence[float], *, tol: float = 0.0) -> bool:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)


def _bboxes_match(
    got: Sequence[float] | None,
    expected: Sequence[float] | None,
    *,
    tol_px: float = 10.0,
) -> bool:
    if got is None or expected is None:
        return got is None and expected is None
    if len(got) < 4 or len(expected) < 4:
        return False
    return all(abs(float(got[i]) - float(expected[i])) <= tol_px for i in range(4))


def _click_point(action: Mapping[str, Any]) -> tuple[float, float] | None:
    if "x" in action and "y" in action:
        return float(action["x"]), float(action["y"])
    box = _bbox_of(action)
    if box is None:
        return None
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class OSWorldAdapter:
    """Bridge OSWorld-like task defs ↔ Dana agent state and step assertions."""

    def __init__(self, *, click_tol_px: float = 10.0, bbox_tol_px: float = 10.0) -> None:
        self.click_tol_px = float(click_tol_px)
        self.bbox_tol_px = float(bbox_tol_px)

    def task_to_agent_state(
        self,
        task: Mapping[str, Any],
        *,
        session_id: str | None = None,
    ) -> ReactGraphState:
        """Map an OSWorld-like JSON task into Dana ``ReactGraphState`` fields."""
        prompt = str(
            task.get("prompt")
            or task.get("instruction")
            or task.get("user_input")
            or ""
        ).strip()
        tid = str(task.get("id") or task.get("task_id") or "osworld-task").strip()
        sid = session_id or f"osworld-{tid}"
        target = task.get("target_screen_state") or task.get("screen_state") or {}
        expected = list(task.get("expected_actions") or task.get("actions") or [])
        env = {
            "benchmark": "osworld",
            "task_id": tid,
            "target_screen_state": dict(target) if isinstance(target, Mapping) else {},
            "expected_actions": expected,
            "elements": list((target or {}).get("elements") or [])
            if isinstance(target, Mapping)
            else [],
        }
        state: ReactGraphState = {
            "session_id": sid,
            "current_agent": "ReAct_Agent",
            "active_intent": prompt or tid,
            "messages": [],
            "iterations": 0,
            "halt": False,
            "fatal_block": False,
            "execution_error": None,
            "critique_history": [],
            "retry_count": 0,
            "max_retries": 3,
            "env_context": env,
            "execution_plan": {
                "source": "osworld_adapter",
                "task_id": tid,
                "steps": expected,
            },
            "plan_index": 0,
            "always_include": list(task.get("always_include") or []),
        }
        # Lazy HumanMessage so adapter imports without langchain in pure unit contexts.
        if prompt:
            try:
                from langchain_core.messages import HumanMessage

                state["messages"] = [HumanMessage(content=prompt)]
            except ImportError:
                state["messages"] = [{"role": "user", "content": prompt}]
                state["last_obs"] = prompt
        return state

    def evaluate_step(
        self,
        action_output: Mapping[str, Any] | None,
        expected_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Score one Dana action against an OSWorld expected step / screen state.

        Returns ``{"passed": bool, "score": float, "reason": str, ...}``.
        Score is 1.0 on full match, partial credit for near-miss clicks / tools.
        """
        got = _norm_action(action_output)
        exp = _norm_action(expected_state)
        if not exp:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "missing expected_state",
                "got": got,
                "expected": exp,
            }
        if not got:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "missing action_output",
                "got": got,
                "expected": exp,
            }

        et = str(exp.get("type") or "")
        gt = str(got.get("type") or "")

        # --- clicks ---
        if et in _CLICK_TYPES or "target" in exp and et in ("", "click"):
            if gt not in _CLICK_TYPES and gt not in ("",):
                # Allow type alias omission when coords present.
                if _click_point(got) is None:
                    return self._fail(got, exp, "action type mismatch (expected click)")
            score, reason = self._score_click(got, exp)
            return {
                "passed": score >= 1.0,
                "score": score,
                "reason": reason,
                "got": got,
                "expected": exp,
            }

        # --- keystrokes / typing ---
        if et in _TYPE_TYPES:
            if gt not in _TYPE_TYPES and "text" not in got and "keys" not in got:
                return self._fail(got, exp, "action type mismatch (expected type/keystroke)")
            want = str(exp.get("text") or exp.get("keys") or exp.get("value") or "")
            have = str(got.get("text") or got.get("keys") or got.get("value") or "")
            if want and have == want:
                return self._ok(got, exp, "keystroke/text match")
            if want and want.lower() in have.lower():
                return {
                    "passed": True,
                    "score": 0.9,
                    "reason": "keystroke substring match",
                    "got": got,
                    "expected": exp,
                }
            return self._fail(got, exp, f"text mismatch: {have!r} != {want!r}")

        # --- tool invocations ---
        if et in _TOOL_TYPES or exp.get("name") or exp.get("tool"):
            want_name = str(exp.get("name") or exp.get("tool") or "").strip()
            have_name = str(got.get("name") or got.get("tool") or "").strip()
            if want_name and have_name and want_name == have_name:
                # Optional args equality (subset).
                want_args = exp.get("args") if isinstance(exp.get("args"), Mapping) else {}
                have_args = got.get("args") if isinstance(got.get("args"), Mapping) else {}
                if want_args:
                    missing = [
                        k
                        for k, v in want_args.items()
                        if have_args.get(k) != v
                    ]
                    if missing:
                        return {
                            "passed": False,
                            "score": 0.5,
                            "reason": f"tool args mismatch keys={missing}",
                            "got": got,
                            "expected": exp,
                        }
                return self._ok(got, exp, f"tool {want_name} match")
            return self._fail(
                got,
                exp,
                f"tool mismatch: {have_name!r} != {want_name!r}",
            )

        # Fallback: structural equality on type + target.
        if gt == et and str(got.get("target") or "") == str(exp.get("target") or ""):
            return self._ok(got, exp, "structural match")
        return self._fail(got, exp, "unrecognized / unmatched action")

    def evaluate_sequence(
        self,
        actions: Sequence[Mapping[str, Any]],
        expected_actions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a full action sequence; mean step score + completion flag."""
        n = len(expected_actions)
        if n == 0:
            return {
                "passed": True,
                "score": 1.0,
                "precision": 1.0,
                "task_completion": 1.0,
                "steps": [],
            }
        steps: list[dict[str, Any]] = []
        for i, exp in enumerate(expected_actions):
            got = actions[i] if i < len(actions) else None
            steps.append(self.evaluate_step(got, exp))
        scores = [float(s["score"]) for s in steps]
        mean = sum(scores) / len(scores)
        completed = sum(1 for s in steps if s["passed"]) / n
        # Precision: among produced actions, fraction that match corresponding expected.
        produced = min(len(actions), n)
        precision = (
            sum(1 for s in steps[:produced] if s["passed"]) / produced if produced else 0.0
        )
        return {
            "passed": all(s["passed"] for s in steps) and len(actions) >= n,
            "score": mean,
            "precision": precision,
            "task_completion": completed,
            "steps": steps,
        }

    def _score_click(
        self, got: Mapping[str, Any], exp: Mapping[str, Any]
    ) -> tuple[float, str]:
        exp_box = _bbox_of(exp)
        got_box = _bbox_of(got)
        # Target label match as soft signal.
        et = str(exp.get("target") or exp.get("element") or "").strip().lower()
        gt = str(got.get("target") or got.get("element") or "").strip().lower()
        label_ok = (not et) or (et and et == gt)

        pt = _click_point(got)
        if exp_box is not None and pt is not None:
            if _point_in_bbox(pt[0], pt[1], exp_box, tol=self.click_tol_px):
                return (1.0, "click inside expected bbox") if label_ok or not et else (
                    0.85,
                    "click inside bbox (label mismatch)",
                )
            # Near miss: center distance.
            cx = (exp_box[0] + exp_box[2]) / 2.0
            cy = (exp_box[1] + exp_box[3]) / 2.0
            dist = ((pt[0] - cx) ** 2 + (pt[1] - cy) ** 2) ** 0.5
            if dist <= self.click_tol_px * 2:
                return 0.5, f"near-miss click dist={dist:.1f}px"
            return 0.0, f"click outside bbox dist={dist:.1f}px"

        if got_box is not None and exp_box is not None:
            if _bboxes_match(got_box, exp_box, tol_px=self.bbox_tol_px):
                return 1.0, "bbox within tolerance"
            return 0.0, "bbox outside tolerance"

        if label_ok and et:
            return 1.0, "target label match"
        return 0.0, "insufficient click geometry"

    @staticmethod
    def _ok(got: dict[str, Any], exp: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "passed": True,
            "score": 1.0,
            "reason": reason,
            "got": got,
            "expected": exp,
        }

    @staticmethod
    def _fail(got: dict[str, Any], exp: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "passed": False,
            "score": 0.0,
            "reason": reason,
            "got": got,
            "expected": exp,
        }
