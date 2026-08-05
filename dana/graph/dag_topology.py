"""Pre-execution DAG solvability checks (entry points, missing deps, cycles)."""

from __future__ import annotations

from collections import deque
from typing import Any


def _normalize_tasks(
    tasks: list[Any],
) -> list[tuple[int, list[int]]]:
    rows: list[tuple[int, list[int]]] = []
    for item in tasks or []:
        if hasattr(item, "task_id"):
            tid = int(item.task_id)
            deps = [int(d) for d in (getattr(item, "dependencies", None) or [])]
        elif isinstance(item, dict):
            tid = int(item["task_id"])
            deps = [int(d) for d in (item.get("dependencies") or [])]
        else:
            raise ValueError(
                f"Topological Error: unsupported task type {type(item)!r}"
            )
        rows.append((tid, deps))
    return rows


def validate_dag_solvability(tasks: list[Any]) -> None:
    """Raise ``ValueError`` if the DAG cannot execute (deadlock / bad topology).

    Checks
    ------
    1. At least one task has ``dependencies: []`` (entry point).
    2. Every dependency id exists in the plan.
    3. No circular dependencies (Kahn topological sort must cover all nodes).
    """
    rows = _normalize_tasks(tasks)
    if not rows:
        raise ValueError("Topological Error: empty task list; no executable DAG.")

    ids = {tid for tid, _ in rows}
    for tid, deps in rows:
        for dep in deps:
            if dep not in ids:
                raise ValueError(
                    f"Topological Error: task {tid} depends on missing "
                    f"prerequisite id {dep} (not in plan)."
                )
            if dep == tid:
                raise ValueError(
                    f"Topological Error: task {tid} cannot depend on itself."
                )

    roots = [tid for tid, deps in rows if not deps]
    if not roots:
        raise ValueError(
            "Topological Error: No starting tasks found. At least one task must "
            "have dependencies: []"
        )

    children: dict[int, list[int]] = {tid: [] for tid, _ in rows}
    indeg: dict[int, int] = {tid: 0 for tid, _ in rows}
    for tid, deps in rows:
        for dep in deps:
            children[dep].append(tid)
            indeg[tid] += 1

    queue: deque[int] = deque(roots)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in children[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    if seen != len(rows):
        cyclic = sorted(tid for tid, deg in indeg.items() if deg > 0)
        raise ValueError(
            "Topological Error: circular dependency detected among task_ids "
            f"{cyclic} (A -> B -> A deadlock)."
        )


__all__ = ("validate_dag_solvability",)
