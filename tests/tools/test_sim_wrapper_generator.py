"""Unit tests for dana.tools.sim_wrapper_generator.generate_simulation_wrapper:
one test per target_platform (isaac_sim/gazebo/ros2/webots) asserting the
generator produces valid, importable-looking boilerplate for each without
raising, plus regression coverage for the two failure paths (unknown
platform, unresolvable urdf_path) and the Workspace Hygiene output-routing
default. Pure text templating, no FreeCAD subprocess/CAD engine involved,
so these are plain pytest functions with real tmp_path files rather than a
mocked path resolver — same "no mock driver needed" convention
tests/tools/test_urdf_builder.py already uses for this same module family.
"""

from __future__ import annotations

import ast
import json
import math
import py_compile

import pytest

from dana.tools.sim_wrapper_generator import _TARGET_PLATFORMS, generate_simulation_wrapper

_MINIMAL_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
</robot>
"""


@pytest.fixture
def urdf_path(tmp_path) -> str:
    path = tmp_path / "test_robot.urdf"
    path.write_text(_MINIMAL_URDF, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("platform", sorted(_TARGET_PLATFORMS))
def test_generates_valid_script_for_every_target_platform(platform, urdf_path, tmp_path) -> None:
    """Every registered target_platform produces ok:True, writes a real
    file, and that file is syntactically valid Python — never a template
    rendering exception, never half-written boilerplate."""
    out_path = tmp_path / f"out_{platform}.py"
    result = json.loads(
        generate_simulation_wrapper("test_assembly", platform, urdf_path, output_path=str(out_path))
    )
    assert result["ok"] is True, result
    assert result["target_platform"] == platform
    assert result["robot_name"] == "test_robot"  # parsed from the URDF's own <robot name=...>
    assert out_path.is_file()
    py_compile.compile(str(out_path), doraise=True)


def test_isaac_sim_script_uses_urdf_importer_kit_command(urdf_path, tmp_path) -> None:
    out_path = tmp_path / "out.py"
    generate_simulation_wrapper("test_assembly", "isaac_sim", urdf_path, output_path=str(out_path))
    content = out_path.read_text(encoding="utf-8")
    assert "SimulationApp" in content
    assert "URDFParseAndImportFile" in content
    assert urdf_path in content or urdf_path.replace("\\", "\\\\") in content


def test_gazebo_script_includes_gazebo_launch_and_spawn(urdf_path, tmp_path) -> None:
    out_path = tmp_path / "out.py"
    generate_simulation_wrapper("test_assembly", "gazebo", urdf_path, output_path=str(out_path))
    content = out_path.read_text(encoding="utf-8")
    assert "gazebo.launch.py" in content
    assert "spawn_entity.py" in content


def test_ros2_script_has_no_gazebo_dependency(urdf_path, tmp_path) -> None:
    out_path = tmp_path / "out.py"
    generate_simulation_wrapper("test_assembly", "ros2", urdf_path, output_path=str(out_path))
    content = out_path.read_text(encoding="utf-8")
    assert "gazebo_ros" not in content
    assert "robot_state_publisher" in content
    assert "joint_state_publisher" in content


def test_webots_script_documents_urdf2webots_conversion_and_discovers_motors(urdf_path, tmp_path) -> None:
    """Webots has no runtime URDF loader (see the generator's own
    docstring) -- the script must document the required offline
    urdf2webots conversion step and use the real, generic device-discovery
    API (Robot.getNumberOfDevices/getDeviceByIndex) rather than a
    hardcoded, fictional one-call import."""
    out_path = tmp_path / "out.py"
    generate_simulation_wrapper("test_assembly", "webots", urdf_path, output_path=str(out_path))
    content = out_path.read_text(encoding="utf-8")
    assert "urdf2webots" in content
    assert "from controller import Node, Supervisor" in content
    assert "getNumberOfDevices" in content
    assert "getDeviceByIndex" in content


def test_unknown_target_platform_rejected(urdf_path, tmp_path) -> None:
    out_path = tmp_path / "out.py"
    result = json.loads(
        generate_simulation_wrapper("test_assembly", "unreal_engine", urdf_path, output_path=str(out_path))
    )
    assert result["ok"] is False
    assert "unreal_engine" in result["error"]
    assert not out_path.exists()


def test_unresolvable_urdf_path_rejected(tmp_path) -> None:
    missing = str(tmp_path / "does_not_exist.urdf")
    result = json.loads(generate_simulation_wrapper("test_assembly", "ros2", missing))
    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_default_output_path_routes_to_sim_wrappers_subdir(urdf_path, tmp_path, monkeypatch) -> None:
    """Workspace Hygiene: with no explicit output_path, the generated
    script lands under a dedicated sim_wrappers/ subdirectory rather than
    as a sibling of the resolved .urdf (the prior default, confirmed live
    to scatter ungitignored *_sim.py files across the repository root
    when a caller's urdf_path happened to resolve into it). Redirects the
    module's own output-base constant into tmp_path first so this never
    touches the real freecad_output/ directory."""
    import dana.tools.sim_wrapper_generator as sim_wrapper_generator

    monkeypatch.setattr(sim_wrapper_generator, "_FREECAD_OUTPUT_DIR", tmp_path / "freecad_output")
    result = json.loads(generate_simulation_wrapper("test_assembly", "ros2", urdf_path))
    assert result["ok"] is True, result
    assert "sim_wrappers" in result["path"].replace("\\", "/").split("/")
    assert str(tmp_path) in result["path"]


# --------------------------------------------------------------------------
# The RL targets' default task logic (reward, termination) — real code, not
# TODO stubs, and runnable without the simulator.
# --------------------------------------------------------------------------

_RL_PLATFORMS = ("isaac_sim", "webots")
_TASK_FUNCTIONS = ("compute_reward", "tilt_from_up_axis_z", "is_terminated")


def _generated_source(platform: str, urdf_path: str, tmp_path) -> str:
    out_path = tmp_path / f"out_{platform}.py"
    result = json.loads(generate_simulation_wrapper("test_assembly", platform, urdf_path, output_path=str(out_path)))
    assert result["ok"] is True, result
    return out_path.read_text(encoding="utf-8")


def _task_defaults(source: str) -> dict:
    """The generated script's task knobs and functions, executed on their
    own: the simulator imports (omni.*, controller) aren't installed here,
    and the defaults must not need them."""
    tree = ast.parse(source)
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _TASK_FUNCTIONS)
        or (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
            and node.targets[0].id in {"TARGET_JOINT_POSITIONS", "ACTION_PENALTY", "MAX_TILT_RAD",
                                       "MIN_HEIGHT_RATIO", "MAX_EPISODE_STEPS"})
    ]
    namespace: dict = {"math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<task-defaults>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("platform", _RL_PLATFORMS)
def test_rl_wrappers_have_no_stubs_and_parse(platform, urdf_path, tmp_path) -> None:
    source = _generated_source(platform, urdf_path, tmp_path)
    assert "TODO" not in source
    tree = ast.parse(source)
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert set(_TASK_FUNCTIONS) <= names
    # step() feeds the defaults and returns Gymnasium's 5-tuple.
    assert "compute_reward(" in source and "is_terminated(" in source
    assert "return observation, reward, terminated, truncated, info" in source or (
        "return self._get_observation(), reward, terminated, truncated, info" in source
    )


@pytest.mark.parametrize("platform", _RL_PLATFORMS)
def test_default_reward_is_finite_and_peaks_on_target(platform, urdf_path, tmp_path) -> None:
    ns = _task_defaults(_generated_source(platform, urdf_path, tmp_path))
    on_target = ns["compute_reward"]([0.1, -0.2], [0.1, -0.2], [0.0, 0.0])
    off_target = ns["compute_reward"]([0.5, 0.4], [0.1, -0.2], [0.0, 0.0])
    with_effort = ns["compute_reward"]([0.1, -0.2], [0.1, -0.2], [2.0, -1.0])
    for value in (on_target, off_target, with_effort):
        assert isinstance(value, float) and math.isfinite(value)
    assert on_target == 0.0
    assert off_target == pytest.approx(-math.hypot(0.4, 0.6))
    assert with_effort == pytest.approx(-ns["ACTION_PENALTY"] * 5.0)


@pytest.mark.parametrize("platform", _RL_PLATFORMS)
def test_default_termination_is_boolean_on_tilt_drop_and_step_limit(platform, urdf_path, tmp_path) -> None:
    ns = _task_defaults(_generated_source(platform, urdf_path, tmp_path))
    done = ns["is_terminated"]
    upright = ns["tilt_from_up_axis_z"](1.0)
    tipped = ns["tilt_from_up_axis_z"](0.0)
    assert upright == 0.0 and tipped == pytest.approx(math.pi / 2)
    assert done(upright, 0.5, 0.5, 1) == (False, False)
    assert done(tipped, 0.5, 0.5, 1) == (True, False)  # fell over
    assert done(upright, 0.2, 0.5, 1) == (True, False)  # dropped below half its height
    assert done(upright, 0.0, 0.0, 1) == (False, False)  # a base on the ground can't "drop"
    assert done(upright, 0.5, 0.5, ns["MAX_EPISODE_STEPS"]) == (False, True)  # time limit
    assert all(isinstance(v, bool) for v in done(tipped, 0.2, 0.5, 10_000))
