"""Offline agent evaluation: five canonical CAD tasks run through Dana's real
ReAct loop (``/ws/chat`` on ``dana.api.server``, in-process via FastAPI's
TestClient) against the trimesh mock CAD engine.

The model is scripted, so a run is deterministic, free and offline, but it
has to get through the same gates a real model does: load the CAD
capability, create a plan, load any tool it isn't offered, then call tools
whose results the real dispatcher validates. What is measured is Dana's
side of the loop:

* **turns**: LLM calls the loop made;
* **tool-schema tokens**: the ``tools=`` payload offered on every call,
  counted the way ``_TOOL_TOKEN_BUDGET`` is measured (cl100k). Each call
  must stay within ``TOOL_TOKEN_CEILINGS`` for its FSM phase; calls over
  the 2,000-token budget itself are counted and reported (see below);
* **retries**: tool calls that failed and were then retried successfully
  (each scenario that scripts a mistake checks the gate actually rejected it);
* **geometry**: assertions on the meshes and tool results, worked out by
  hand from the dimensions asked for.

Offline is enforced, not assumed: while a run is in progress, subprocesses
(FreeCADCmd, a slicer) and any non-loopback network connection raise, and
each attempt is counted; a passing run has none.

Used by ``tests/evals/test_offline_evals.py`` (inside the normal suite) and
``scripts/run_offline_evals.py`` (prints the markdown scorecard).
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Patch = Callable[[Any, str, Any], None]
SetEnv = Callable[[str, str], None]

MODEL = "eval/scripted-model"

# The largest tools= payload (cl100k tokens) each FSM phase offers across
# these scenarios: regression ceilings, ratcheted down whenever a schema
# shrinks and never raised to make a run pass. History: Phase 5 measured
# planning 2,826 / executing 3,382 / done 10,336 (the wrap-up turn offered
# the whole freecad_full domain, more than a local model's 8,192-token
# context); Phase 6 capped the wrap-up set, dropped plan-dependent tools
# where they can only fail and tightened the core descriptions.
# PHASE_TARGETS are the hard limits the ceilings must stay within.
PHASE_TARGETS = {"planning": 2000, "executing": 3000, "done": 2000}
# planning and executing are fixed allow-lists, so their ceilings are exact.
# The wrap-up set fills optional tools by relevance until _TOOL_TOKEN_BUDGET
# runs out (1,940-1,995 here), so its ceiling is the budget itself.
TOOL_TOKEN_CEILINGS = {"planning": 1476, "executing": 2559, "done": 2000}
_ENV = {"DANA_HEADLESS": "true", "DANA_OS_DRY_RUN": "1", "DANA_CONTEXT_DISTILL": "0", "DANA_AUTO_APPROVE": "0"}
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class Step:
    tool_id: str
    args: dict[str, Any]
    expect_error: bool = False  # a deliberate mistake the gate must reject


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Scenario:
    key: str
    title: str
    prompt: str
    tasks: list[dict[str, Any]]
    steps: list[Step]
    reply: str
    checks: Callable[[Run], list[Check]]
    # Runs after the chat turn on the same socket (Task 5's rebuild).
    after: Callable[[Any, Run], None] | None = None


@dataclass
class Run:
    scenario: Scenario
    session_id: str = ""
    llm_calls: int = 0
    tool_tokens: list[int] = field(default_factory=list)
    tool_phases: list[str] = field(default_factory=list)  # the FSM phase of each call
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    final_reply: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    geometry: list[Check] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None
    model: Any = None

    @property
    def max_tool_tokens(self) -> int:
        return max(self.tool_tokens, default=0)

    @property
    def turns_over_budget(self) -> int:
        from dana.core import react_dispatch as rd

        return sum(t > rd._TOOL_TOKEN_BUDGET for t in self.tool_tokens)

    def ceiling_breaches(self) -> list[str]:
        return [
            f"{phase} turn {i + 1}: {tokens} > {TOOL_TOKEN_CEILINGS.get(phase)}"
            for i, (phase, tokens) in enumerate(zip(self.tool_phases, self.tool_tokens))
            if tokens > TOOL_TOKEN_CEILINGS.get(phase, 0)
        ]

    @property
    def retries(self) -> int:
        """Failed tool calls followed later by a successful call of the same tool."""
        count = 0
        for i, event in enumerate(self.tool_events):
            if event["status"] != "success" and any(
                later["tool_id"] == event["tool_id"] and later["status"] == "success"
                for later in self.tool_events[i + 1:]
            ):
                count += 1
        return count

    @property
    def passed(self) -> bool:
        return self.error is None and not self.blocked and all(c.passed for c in self.checks)

    def outputs(self, tool_id: str) -> list[dict[str, Any]]:
        return [e["output"] or {} for e in self.tool_events if e["tool_id"] == tool_id and e["status"] == "success"]

    def last_output(self, tool_id: str) -> dict[str, Any]:
        found = self.outputs(tool_id)
        return found[-1] if found else {}

    def mesh(self, name: str) -> Any:
        """The mock engine's current mesh for ``name`` in this session."""
        import trimesh

        from dana.platform import mock

        return trimesh.load(mock._MOCK_OBJECT_REGISTRY[self.session_id][name], force="mesh")


# --- the scripted model ---------------------------------------------------------------


def _offered(tools: Any) -> set[str]:
    return {(t.get("function") or {}).get("name") or t.get("name") for t in tools or [] if isinstance(t, dict)}


class ScriptedModel:
    """Stands in for ``ModelProvider``: plays ``scenario``'s calls in order,
    loading any tool it isn't offered first, then replies."""

    def __init__(self, run: Run) -> None:
        self._run = run
        scenario = run.scenario
        self._sequence = [
            Step("load_capability", {"domain": "freecad_full"}),
            Step("create_plan", {"objective": scenario.prompt, "tasks": scenario.tasks}),
            *scenario.steps,
        ]
        self._next = 0
        self._loaded: set[str] = set()

    def complete_with_tool_calls(self, messages: Any, *, tools: Any, provider: Any = None, **kwargs: Any) -> dict:
        from dana.core import react_dispatch as rd
        from dana.tools.schema import ToolCall

        self._run.llm_calls += 1
        self._run.tool_tokens.append(rd._count_tokens(json.dumps(tools or [])))
        sid = self._run.session_id
        self._run.tool_phases.append(rd._get_fsm_state(sid) if rd._get_has_plan(sid) else "planning")
        result = {"model": MODEL, "provider": "eval", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        if self._next < len(self._sequence):
            step = self._sequence[self._next]
            gated = step.tool_id not in ("load_capability", "create_plan")
            if gated and step.tool_id not in _offered(tools) and step.tool_id not in self._loaded:
                self._loaded.add(step.tool_id)
                return {**result, "content": "", "tool_calls": [ToolCall("load_specific_tool", {"tool_id": step.tool_id})]}
            self._next += 1
            return {**result, "content": "", "tool_calls": [ToolCall(step.tool_id, dict(step.args))]}
        return {**result, "content": self._run.scenario.reply, "tool_calls": []}


# --- environment ----------------------------------------------------------------------


def install(patch: Patch, setenv: SetEnv, run_ref: list[Run]) -> None:
    """Points the server at the mock engine and the scripted model, and
    blocks subprocesses and non-loopback network access. ``patch``/``setenv``
    are ``monkeypatch.setattr``/``setenv`` under pytest (undone after each
    test) or plain setters in the scorecard script. ``run_ref[0]`` is the
    run in progress."""
    import dana.api.server as server
    import dana.core.model_provider as model_provider
    import dana.core.react_dispatch as rd
    from dana.platform import factory
    from dana.platform.mock import MockControlPlane, MockFreeCADEngine
    from dana.plugins.freecad import engine

    for key, value in _ENV.items():
        setenv(key, value)
    # Load tiktoken's encoding now, before the network guard below: on a
    # cold cache it downloads, and blocked it would leave _count_tokens on
    # its len//3 estimate for the rest of the process.
    rd._count_tokens("warm-up")
    patch(model_provider, "ensure_dotenv_loaded", lambda: None)
    patch(server, "get_cad_engine", lambda: MockFreeCADEngine())
    # Plugin tools (run_full_manufacturing_pipeline) get their engine from
    # the factory rather than from the server.
    patch(factory, "get_cad_engine", lambda: MockFreeCADEngine())
    patch(server, "get_control_plane", lambda: MockControlPlane())
    patch(server, "_default_auto_approve", lambda: True)
    # ModelProvider is constructed afresh for every LLM call; the script's
    # position lives on the run.
    patch(rd, "ModelProvider", lambda **_k: run_ref[0].model)

    def no_freecad(*_a: Any, **_k: Any) -> dict[str, Any]:
        run_ref[0].extra.setdefault("freecad_calls_refused", 0)
        run_ref[0].extra["freecad_calls_refused"] += 1
        return {"ok": False, "error": "offline eval: FreeCADCmd is not available", "stdout": "", "stderr": ""}

    def no_subprocess(*args: Any, **_k: Any) -> Any:
        run_ref[0].blocked.append(f"subprocess {args[0] if args else '?'!r}")
        raise OSError("offline eval: subprocesses are disabled")

    real_connect = socket.socket.connect

    def loopback_only(sock: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            run_ref[0].blocked.append(f"network {host}")
            raise OSError("offline eval: network access is disabled")
        return real_connect(sock, address)

    patch(engine, "_run_freecad_script", no_freecad)
    patch(subprocess, "Popen", no_subprocess)
    patch(socket.socket, "connect", loopback_only)


@contextmanager
def standalone_environment(workspace: Path) -> Iterator[list[Run]]:
    """``install`` outside pytest: the workspace redirected to ``workspace``
    (tests/workspace_isolation.py), everything restored on exit."""
    import dana.api.server  # noqa: F401 — loaded before the sweep, like everything it imports
    import dana.plugins.os.file_system as file_system
    from workspace_isolation import import_path_holders, workspace_redirects

    undo: list[tuple[Any, str, Any]] = []
    env_before = {k: os.environ.get(k) for k in _ENV}

    def patch(obj: Any, name: str, value: Any) -> None:
        undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def setenv(key: str, value: str) -> None:
        os.environ[key] = value

    import_path_holders()
    for module, attr, value in list(workspace_redirects(workspace)):
        patch(module, attr, value)
    Path(file_system._SANDBOX_ROOT).mkdir(parents=True, exist_ok=True)
    run_ref: list[Run] = []
    install(patch, setenv, run_ref)
    try:
        yield run_ref
    finally:
        for obj, name, value in reversed(undo):
            setattr(obj, name, value)
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- driving one scenario --------------------------------------------------------------


def _receive_until(ws: Any, run: Run, wanted: str, limit: int = 400) -> dict[str, Any]:
    for _ in range(limit):
        event = ws.receive_json()
        if event.get("type") == "tool_dispatch_end":
            run.tool_events.append(event)
        if event.get("type") == wanted:
            return event
    raise AssertionError(f"no {wanted!r} event after {limit} messages")


def run_scenario(scenario: Scenario, run_ref: list[Run]) -> Run:
    """Runs ``scenario`` on a fresh /ws/chat session and evaluates its checks."""
    from fastapi.testclient import TestClient

    import dana.api.server as server

    run = Run(scenario)
    run.model = ScriptedModel(run)
    run_ref[:] = [run]
    started = time.perf_counter()
    try:
        # Not entered as a context manager: that would run the server's
        # lifespan (voice service, Whisper load, printer watchdog, stdout
        # broadcast), none of which a chat turn needs.
        client = TestClient(server.app)
        with client.websocket_connect("/ws/chat") as ws:
            run.session_id = _receive_until(ws, run, "ready")["session_id"]
            ws.send_json({"type": "update_context", "active_plugins": ["freecad"]})
            ws.send_json({"text": scenario.prompt})
            run.final_reply = _receive_until(ws, run, "assistant_message")["content"]
            if scenario.after is not None:
                scenario.after(ws, run)
            run.geometry = scenario.checks(run)
            run.checks = _common_checks(run) + run.geometry
    except Exception as exc:  # noqa: BLE001 — reported as a failed task, not a crashed suite
        run.error = f"{type(exc).__name__}: {exc}"
    run.duration_s = time.perf_counter() - started
    return run


def _common_checks(run: Run) -> list[Check]:
    from dana.core import react_dispatch as rd

    scripted = [s for s in run.scenario.steps if not s.expect_error]
    mistakes = [s for s in run.scenario.steps if s.expect_error]
    succeeded = sum(1 for e in run.tool_events if e["status"] == "success" and e["tool_id"] not in (
        "load_capability", "create_plan", "load_specific_tool"))
    return [
        Check("final reply", run.final_reply == run.scenario.reply, run.final_reply[:80]),
        Check("every scripted call succeeded", succeeded >= len(scripted), f"{succeeded}/{len(scripted)}"),
        Check("scripted mistakes were rejected and retried", run.retries == len(mistakes), f"retries={run.retries}"),
        Check(
            "tool schema within its phase's ceiling",
            len(run.tool_phases) == run.llm_calls and not run.ceiling_breaches(),
            "; ".join(run.ceiling_breaches()) or f"max={run.max_tool_tokens}",
        ),
        Check(f"_TOOL_TOKEN_BUDGET is {rd._TOOL_TOKEN_BUDGET} (<= 2000)", rd._TOOL_TOKEN_BUDGET <= 2000),
        Check(
            "phase ceilings within their targets",
            all(TOOL_TOKEN_CEILINGS[phase] <= limit for phase, limit in PHASE_TARGETS.items()),
            str(TOOL_TOKEN_CEILINGS),
        ),
    ]


def _close(name: str, actual: float, expected: float, rel: float = 1e-6, abs_tol: float = 1e-6) -> Check:
    ok = actual is not None and math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol)
    return Check(name, ok, f"{actual!r} vs {expected!r}")


def _extents(mesh: Any) -> list[float]:
    return [float(v) for v in mesh.extents]


# A trimesh cylinder is a 32-gon: its area is this fraction of the circle's.
_POLYGON_32 = (32 / (2 * math.pi)) * math.sin(2 * math.pi / 32)


# --- the five canonical tasks -----------------------------------------------------------


def _task1_checks(run: Run) -> list[Check]:
    box, rod = run.outputs("query_topology")[:2] if len(run.outputs("query_topology")) >= 2 else ({}, {})
    bbox = box.get("bounding_box") or {}
    return [
        Check("box: 6 planar faces", box.get("face_count") == 6 and all(f["is_planar"] for f in box.get("faces", []))),
        Check(
            "box: bounding box 30 x 20 x 10",
            [bbox.get("xlen"), bbox.get("ylen"), bbox.get("zlen")] == [30.0, 20.0, 10.0],
            str(bbox),
        ),
        _close("box: volume 6000 mm³", box.get("volume"), 6000.0),
        Check(
            "cylinder: 2 caps + 1 curved side",
            [f["is_planar"] for f in rod.get("faces", [])] == [True, True, False],
        ),
        _close("cylinder: height 25", (rod.get("bounding_box") or {}).get("zlen"), 25.0),
        _close("cylinder: volume π·8²·25 (32-gon)", rod.get("volume"), math.pi * 64 * 25 * _POLYGON_32, rel=1e-3),
    ]


_PLATE = Step("create_freecad_box", {"name": "Plate", "length": 80, "width": 50, "height": 6})
_HOLE_SPOTS = [(-30, -15), (30, -15), (30, 15), (-30, 15)]


def _hole_steps() -> list[Step]:
    steps = [
        # A plausible model mistake: "upper" isn't a face name. The tool's
        # enum check rejects it and the model retries with "top".
        Step(
            "create_freecad_feature_on_face",
            {"object_name": "Plate", "face": "upper", "shape": "circle", "u": -30, "v": -15, "extent": 6,
             "operation": "cut", "radius": 3, "name": "Hole1"},
            expect_error=True,
        )
    ]
    base = "Plate"
    for i, (u, v) in enumerate(_HOLE_SPOTS, start=1):
        steps.append(
            Step(
                "create_freecad_feature_on_face",
                {"object_name": base, "face": "top", "shape": "circle", "u": u, "v": v, "extent": 6,
                 "operation": "cut", "radius": 3, "name": f"Hole{i}"},
            )
        )
        base = f"Hole{i}"
    return steps


_TASK2_STEPS = [
    _PLATE,
    *_hole_steps(),
    Step("perform_freecad_edge_operation", {"operation": "chamfer", "target_object": "Hole4", "value": 1}),
    Step("query_topology", {"part_name": "Chamfer"}),
]
# One task per operation: create_plan rejects a geometry task that groups
# several ("four holes"), so a real model has to split it this way too.
_TASK2_TASKS = [
    {"description": "Create the base plate", "expected_tools": ["create_freecad_box"]},
    *(
        {"description": f"Cut mounting hole {i} through the top face", "expected_tools": ["create_freecad_feature_on_face"]}
        for i in range(1, 5)
    ),
    {"description": "Chamfer the plate's edges", "expected_tools": ["perform_freecad_edge_operation"]},
    {"description": "Inspect the finished plate", "expected_tools": ["query_topology"]},
]
_HOLE_VOLUME = math.pi * 9 * 6 * _POLYGON_32


def _task2_checks(run: Run) -> list[Check]:
    topo = run.last_output("query_topology")
    faces = topo.get("faces", [])
    top = next((f for f in faces if f.get("direction") == "+Z"), {})
    bores = [f for f in faces if not f["is_planar"]]
    plate = run.mesh("Chamfer")
    return [
        Check("four through-holes (curved bores)", len(bores) == 4, f"{len(bores)} curved faces"),
        _close("top face lost 4 hole areas", top.get("area"), 80 * 50 - 4 * math.pi * 9 * _POLYGON_32, rel=1e-3),
        _close("volume = plate - 4 holes", float(plate.volume), 80 * 50 * 6 - 4 * _HOLE_VOLUME, rel=1e-3),
        Check("outer size unchanged 80 x 50 x 6", _extents(plate) == [80.0, 50.0, 6.0], str(_extents(plate))),
        Check("hole centers on the top face", all(
            any(math.isclose(b["centroid"][0], u, abs_tol=0.05) and math.isclose(b["centroid"][1], v, abs_tol=0.05)
                for b in bores) for u, v in _HOLE_SPOTS)),
    ]


def _task3_checks(run: Run) -> list[Check]:
    block = run.mesh("Bored")
    planning = [t for p, t in zip(run.tool_phases, run.tool_tokens) if p == "planning"]
    return [
        _close("cut volume = block - bore", float(block.volume), 40 * 40 * 20 - math.pi * 64 * 20 * _POLYGON_32, rel=1e-3),
        Check("result is a closed solid", bool(block.is_watertight)),
        Check("bbox unchanged by the cut", _extents(block) == [40.0, 40.0, 20.0], str(_extents(block))),
        Check(
            "every turn measured, each within its phase's ceiling",
            len(run.tool_tokens) == run.llm_calls and not run.ceiling_breaches(),
            f"{run.llm_calls} turns, {run.turns_over_budget} over the 2,000 budget, max {run.max_tool_tokens}",
        ),
        Check(
            "planning turns offer the core tool set only",
            bool(planning) and max(planning) <= TOOL_TOKEN_CEILINGS["planning"],
            f"{len(planning)} planning turns, max {max(planning, default=0)}",
        ),
    ]


def _assembly_steps(post_z: float) -> list[Step]:
    return [
        Step("create_freecad_box", {"name": "Base", "length": 40, "width": 40, "height": 10}),
        Step("create_freecad_cylinder", {"name": "Post", "radius": 5, "height": 30, "placement_z": post_z}),
        Step("create_freecad_assembly", {"name": "Stand"}),
        Step("add_parts_to_assembly", {"assembly_name": "Stand", "part_names": ["Base", "Post"]}),
    ]


def _task4_checks(run: Run) -> list[Check]:
    pipeline = run.last_output("run_full_manufacturing_pipeline")
    collisions = run.outputs("validate_assembly_collisions")
    bom = run.last_output("generate_assembly_bom")
    prints = run.outputs("check_printability")
    spaced = collisions[-1] if collisions else {}
    masses = {p["name"]: p["mass_g"] for p in bom.get("parts", [])}
    return [
        Check("colliding stand: pipeline complete is False", pipeline.get("complete") is False, str(pipeline.get("complete"))),
        Check(
            "colliding stand: blamed on the Base/Post collision",
            "parts collide: Base/Post" in (pipeline.get("failure_reason") or ""),
            (pipeline.get("failure_reason") or "")[:120],
        ),
        Check("spaced stand: collision-free", spaced.get("has_collisions") is False and spaced.get("collision_free") is True),
        _close("spaced stand: 1 mm clearance", ((spaced.get("clearances") or [{}])[0]).get("distance_mm"), 1.0, abs_tol=1e-4),
        _close("BOM: Base 16 cm³ of PLA = 19.84 g", masses.get("Base"), 19.84, abs_tol=0.01),
        _close("BOM: Post π·5²·30 mm³ of PLA", masses.get("Post"), math.pi * 25 * 30 * _POLYGON_32 / 1000 * 1.24, rel=1e-3),
        Check("both parts printable on the MK4", len(prints) == 2 and all(p.get("printable") for p in prints)),
        *_urdf_checks(run.last_output("export_assembly_to_urdf")),
    ]


def _urdf_checks(result: dict[str, Any]) -> list[Check]:
    import xml.etree.ElementTree as ET

    path = Path(result.get("path") or "")
    if not path.is_file():
        return [Check("URDF exported", False, str(result.get("error") or result)[:120])]
    root = ET.parse(path).getroot()
    links = root.findall("link")
    meshes = [path.parent / m.get("filename") for m in root.iter("mesh")]
    return [
        Check("URDF exported", root.tag == "robot", path.name),
        Check("URDF links: base_link, Base, Post", sorted(link.get("name") for link in links) == ["Base", "Post", "base_link"],
              str([link.get("name") for link in links])),
        Check("every link mesh exists", bool(meshes) and all(m.is_file() for m in meshes), f"{len(meshes)} meshes"),
        Check("links carry inertial mass", all(link.find("inertial/mass") is not None for link in links
                                               if link.get("name") != "base_link")),
    ]


def _rebuild(ws: Any, run: Run) -> None:
    """Task 5: read the Parameters table, change the plate's length, rebuild."""
    llm_calls = run.llm_calls
    ws.send_json({"type": "list_parameters"})
    table = _receive_until(ws, run, "parameters")
    length = next(p for p in table["params"] if p["label"] == "Plate.length")
    run.extra["parameter_count"] = len(table["params"])
    ws.send_json({"type": "rebuild", "overrides": {length["id"]: 100}})
    run.extra["rebuild"] = _receive_until(ws, run, "rebuild_result")
    run.extra["rebuild_llm_calls"] = run.llm_calls - llm_calls


def _task5_checks(run: Run) -> list[Check]:
    rebuild = run.extra.get("rebuild") or {}
    plate = run.mesh("Chamfer")
    return [
        Check("rebuild succeeded", rebuild.get("ok") is True, str(rebuild.get("error"))),
        Check("0 LLM calls during the rebuild", run.extra.get("rebuild_llm_calls") == 0, str(run.extra.get("rebuild_llm_calls"))),
        Check("counted as a zero-token rebuild", rebuild.get("zero_token_rebuilds") == 1),
        Check("rebuilt plate is 100 x 50 x 6", _extents(plate) == [100.0, 50.0, 6.0], str(_extents(plate))),
        _close("rebuilt volume = 100·50·6 - 4 holes", float(plate.volume), 100 * 50 * 6 - 4 * _HOLE_VOLUME, rel=1e-3),
    ]


SCENARIOS: list[Scenario] = [
    Scenario(
        key="t1_primitive_topology",
        title="Single primitive + topology check",
        prompt="Make a 30 x 20 x 10 mm block and an 8 mm radius, 25 mm tall rod, and confirm their dimensions.",
        tasks=[
            {"description": "Create the block", "expected_tools": ["create_freecad_box"]},
            {"description": "Measure the block", "expected_tools": ["query_topology"]},
            {"description": "Create the rod", "expected_tools": ["create_freecad_cylinder"]},
            {"description": "Measure the rod", "expected_tools": ["query_topology"]},
        ],
        steps=[
            Step("create_freecad_box", {"name": "Block", "length": 30, "width": 20, "height": 10}),
            Step("query_topology", {"part_name": "Block"}),
            Step("create_freecad_cylinder", {"name": "Rod", "radius": 8, "height": 25}),
            Step("query_topology", {"part_name": "Rod"}),
        ],
        reply="The block is 30 x 20 x 10 mm and the rod is 8 mm in radius and 25 mm tall.",
        checks=_task1_checks,
    ),
    Scenario(
        key="t2_multi_feature",
        title="Multi-feature part (holes + chamfer)",
        prompt="An 80 x 50 x 6 mm mounting plate with four 6 mm holes on the top face and chamfered edges.",
        tasks=_TASK2_TASKS,
        steps=_TASK2_STEPS,
        reply="The mounting plate has four 6 mm through-holes and chamfered edges.",
        checks=_task2_checks,
    ),
    Scenario(
        key="t3_boolean_budget",
        title="Boolean cut + tool-token budget",
        prompt="Bore a 16 mm hole straight through a 40 x 40 x 20 mm block.",
        tasks=[
            {"description": "Create the block", "expected_tools": ["create_freecad_box"]},
            {"description": "Create the bore cylinder", "expected_tools": ["create_freecad_cylinder"]},
            {"description": "Cut the bore out of the block", "expected_tools": ["perform_freecad_boolean"]},
        ],
        steps=[
            Step("create_freecad_box", {"name": "Block", "length": 40, "width": 40, "height": 20}),
            Step("create_freecad_cylinder", {"name": "Bore", "radius": 8, "height": 40}),
            Step("perform_freecad_boolean", {"operation": "cut", "base_object": "Block", "tool_object": "Bore",
                                             "name": "Bored"}),
        ],
        reply="The block now has a 16 mm through-bore.",
        checks=_task3_checks,
    ),
    Scenario(
        key="t4_assembly_verdict",
        title="Assembly + manufacturing verdict",
        prompt="Build a stand (40 mm base, 10 mm post), check it for manufacture, then fix the post and verify it.",
        tasks=[
            {"description": "Create the base", "expected_tools": ["create_freecad_box"]},
            {"description": "Create the post", "expected_tools": ["create_freecad_cylinder"]},
            {"description": "Assemble the stand", "expected_tools": ["create_freecad_assembly", "add_parts_to_assembly"]},
            {"description": "Run the manufacturing pipeline", "expected_tools": ["run_full_manufacturing_pipeline"]},
            {"description": "Lift the post clear of the base",
             "expected_tools": ["get_freecad_bounding_box", "position_assembly_part"]},
            {"description": "Re-check the stand for collisions", "expected_tools": ["validate_assembly_collisions"]},
            {"description": "Export the stand as a URDF", "expected_tools": ["export_assembly_to_urdf"]},
            {"description": "Cost the stand", "expected_tools": ["generate_assembly_bom"]},
            {"description": "Check the base prints", "expected_tools": ["check_printability"]},
            {"description": "Check the post prints", "expected_tools": ["check_printability"]},
        ],
        steps=[
            # The post is centred on the base, so the two overlap.
            *_assembly_steps(post_z=0),
            Step("run_full_manufacturing_pipeline", {"assembly_name": "Stand"}),
            # Moving a part before measuring its reference is blocked by the
            # bounding-box gate; the model measures, then retries. Lifted to
            # sit 1 mm above the base: base top z=5, post spans 21±15.
            Step("position_assembly_part", {"part_name": "Post", "placement_z": 21}, expect_error=True),
            Step("get_freecad_bounding_box", {"target_object": "Base"}),
            Step("position_assembly_part", {"part_name": "Post", "placement_z": 21}),
            Step("validate_assembly_collisions", {"assembly_name": "Stand"}),
            Step("export_assembly_to_urdf", {"assembly_name": "Stand"}),
            Step("generate_assembly_bom", {"assembly_name": "Stand"}),
            Step("check_printability", {"object_name": "Base"}),
            Step("check_printability", {"object_name": "Post"}),
        ],
        reply="The first stand collided; with the post lifted it is collision-free, costed and printable.",
        checks=_task4_checks,
    ),
    Scenario(
        key="t5_zero_token_rebuild",
        title="Zero-token parametric rebuild",
        prompt="An 80 x 50 x 6 mm mounting plate with four 6 mm holes on the top face and chamfered edges.",
        tasks=_TASK2_TASKS,
        steps=_TASK2_STEPS,
        reply="The mounting plate has four 6 mm through-holes and chamfered edges.",
        checks=_task5_checks,
        after=_rebuild,
    ),
]


def scorecard(runs: list[Run]) -> str:
    """Markdown table of ``runs``."""
    lines = [
        "| Task | Result | Turns | Max tool-schema tokens | Turns over 2,000 | Retries | Geometry checks | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        geometry = run.geometry
        result = "PASS" if run.passed else "FAIL"
        lines.append(
            f"| {run.scenario.title} | {result} | {run.llm_calls} | {run.max_tool_tokens} | "
            f"{run.turns_over_budget}/{run.llm_calls} | {run.retries} | "
            f"{sum(c.passed for c in geometry)}/{len(geometry)} | {run.duration_s:.2f} s |"
        )
    passed = sum(r.passed for r in runs)
    lines.append("")
    lines.append(f"**{passed}/{len(runs)} tasks passed.**")
    lines.append("")
    lines.append(
        "Tool-schema tokens are held to per-phase ceilings "
        + ", ".join(f"{phase} {limit:,}" for phase, limit in TOOL_TOKEN_CEILINGS.items())
        + " (targets: planning and post-plan within the 2,000-token budget, executing under 3,000)."
    )
    for run in runs:
        failures = [c for c in run.checks if not c.passed]
        if run.error or failures or run.blocked:
            lines.append("")
            lines.append(f"{run.scenario.title}:")
            if run.error:
                lines.append(f"- error: {run.error}")
            lines.extend(f"- blocked: {b}" for b in run.blocked)
            lines.extend(f"- failed: {c.name} ({c.detail})" for c in failures)
    return "\n".join(lines)
