"""Simulation-loader script generator — the RL Pipeline bridge's last mile.

``dana.plugins.freecad.engine.export_assembly_to_urdf``/``dana.tools.
urdf_builder`` produce a physics-ready ``.urdf`` (real collision/inertial
data, a real kinematic tree); this module writes the standalone Python
boilerplate that actually LOADS that ``.urdf`` into a simulator —
Isaac Sim, a ROS2/Gazebo launch file, or a Webots controller — so a
caller doesn't have to hand-write the loader every time. Pure text
templating, no FreeCAD subprocess or CAD engine involved (same "no
get_cad_engine() abstraction needed" reasoning as ``dana.tools.
urdf_builder``'s own module docstring).

Every public function returns a JSON string (``{"ok": bool, ...}``), same
wire contract as every ``dana.plugins.freecad.engine``/``dana.tools.
urdf_builder`` function, so it slots into ``dana.core.react_dispatch``'s
handler dict the same way.

The generated scripts are boilerplate/stubs, not a finished RL pipeline —
the task this bridges to (Isaac Sim's own Python API surface across
versions, a real reward/termination function, a real ROS2 package layout,
a real Webots PROTO conversion) is inherently something only the caller's
own project can finish; see each generator function's own docstring for
exactly what's stubbed and why.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dana.paths import DANA_WORKSPACE
from dana.session_context import session_scoped_dir

# Same two base directories dana.plugins.freecad.engine/dana.tools.urdf_builder
# already write generated artifacts to — declared as plain copies rather
# than importing either module's underscore-prefixed attribute across
# modules (same precedent urdf_builder.py's own _OUTPUT_DIR docstring
# already applies). Used only to RESOLVE a bare urdf_path filename a caller
# gives without its full directory (see _resolve_urdf_path).
_FREECAD_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"
_EXPORT_DIR = DANA_WORKSPACE / "exports"

# Workspace Hygiene: every generated wrapper script is written under this
# session-scoped subdirectory (see generate_simulation_wrapper's own
# out_path default) rather than as a sibling of the resolved .urdf —
# confirmed live that the sibling-of-the-urdf default let repeated manual
# testing scatter ungitignored *_sim.py/launch files directly into the
# repository root (whatever directory a caller's urdf_path happened to
# resolve into that session), with no single predictable location to find
# or clean them up from afterward.
_SIM_WRAPPERS_SUBDIR = "sim_wrappers"

_TARGET_PLATFORMS = frozenset({"isaac_sim", "gazebo", "ros2", "webots"})


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": str(message)})


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name or "").strip("_") or "robot"


def _resolve_urdf_path(urdf_path: str) -> Path | None:
    """Poll a few candidate locations for a caller-supplied ``urdf_path``
    before ever reading/embedding it — same "guard against a hallucinated
    path" reasoning as ``dana.tools.urdf_builder._mesh_file_exists``: an
    LLM can freely invent a plausible-looking ``urdf_path`` that was never
    actually produced by ``export_assembly_to_urdf``/``generate_urdf_assembly``.

    Checked in order: the path as given (absolute, or already
    cwd-relative-and-correct); then its basename under THIS session's own
    ``freecad_output/`` dir (where ``generate_urdf_assembly`` writes);
    then under ``exports/`` and each of its immediate subdirectories
    (where ``export_assembly_to_urdf`` writes, one ``<name>_urdf/``
    subfolder per assembly); then its basename under the process's
    current working directory.
    """
    candidate = Path(urdf_path)
    if candidate.is_file():
        return candidate
    basename = candidate.name

    freecad_output_dir = session_scoped_dir(_FREECAD_OUTPUT_DIR)
    direct = freecad_output_dir / basename
    if direct.is_file():
        return direct

    export_dir = session_scoped_dir(_EXPORT_DIR)
    direct = export_dir / basename
    if direct.is_file():
        return direct
    if export_dir.is_dir():
        for sub in export_dir.iterdir():
            if sub.is_dir():
                nested = sub / basename
                if nested.is_file():
                    return nested

    cwd_candidate = Path.cwd() / basename
    if cwd_candidate.is_file():
        return cwd_candidate
    return None


def _robot_name_from_urdf(urdf_path: Path, fallback: str) -> str:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return fallback
    return root.get("name") or fallback


def _isaac_sim_script(assembly_name: str, robot_name: str, urdf_path: Path, output_filename: str) -> str:
    """Standalone Isaac Sim loader: boots ``SimulationApp``, imports the
    URDF via the URDF importer extension's own stable ``omni.kit.commands``
    entry point (``"URDFParseAndImportFile"`` — the one part of the URDF
    importer's Python surface that has stayed name-stable across the
    ``omni.isaac.urdf`` -> ``omni.importer.urdf`` -> ``isaacsim.asset.
    importer.urdf`` extension-path renames between Isaac Sim releases;
    the raw ``_urdf`` module path below is the one thing a caller may need
    to adjust for their installed version), spawns it at the origin on a
    default ground plane, and wraps it in a minimal Gym-style env class
    with ``reset()``/``step()`` stubs operating on PyTorch tensors — the
    RL-pipeline half this whole tool exists to hand off to a caller's own
    project, deliberately left unimplemented (a real reward/termination
    function is a modeling decision only the caller can make).
    """
    class_name = "".join(w[0].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", assembly_name) if w) or "Robot"
    # repr(), not manual "..."/r"..." wrapping: robot_name comes straight
    # from the URDF's own <robot name="..."> attribute (ElementTree already
    # un-escapes any &quot;/&amp; in it), so a literal `"` or `\` in there
    # would otherwise break the ROBOT_NAME/URDF_PATH assignment below —
    # repr() always produces a syntactically valid, safely-escaped literal
    # regardless of what either string actually contains.
    urdf_path_repr = repr(str(urdf_path))
    robot_name_repr = repr(robot_name)
    return f'''"""Auto-generated Isaac Sim loader for '{assembly_name}' — produced by
Dana's generate_simulation_wrapper tool from {urdf_path.name}.

Run with Isaac Sim's OWN bundled Python interpreter (not a system Python):
    <isaac-sim-install-dir>/python.sh {output_filename}     # Linux
    <isaac-sim-install-dir>/python.bat {output_filename}    # Windows

Requires a local Isaac Sim installation. The `omni.importer.urdf` import
below matches Isaac Sim 2023.1.x/4.0.x; older installs use
`omni.isaac.urdf`, newer ones `isaacsim.asset.importer.urdf` — adjust that
one import line if this doesn't match your installed version (the
"URDFParseAndImportFile" kit command name itself has stayed stable across
all three).
"""

from __future__ import annotations

# SimulationApp MUST be constructed before any other omni.* import below —
# it's what actually loads the Kit application these modules hook into.
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({{"headless": False}})

import numpy as np
import torch
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.kit.commands
from omni.importer.urdf import _urdf

URDF_PATH = {urdf_path_repr}
ROBOT_NAME = {robot_name_repr}
ROBOT_PRIM_PATH = f"/World/{{ROBOT_NAME}}"


def _import_urdf_as_usd(urdf_path: str) -> str:
    """Runs the URDF importer extension's own stable kit command and
    returns the resulting USD stage path it produced. `fix_base=True`
    anchors the robot's root link in place (set False for a mobile/legged
    base); `import_inertial_tensor=True` reuses '{urdf_path.name}''s own
    real <inertial> data instead of re-deriving one from the mesh."""
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.import_inertia_tensor = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    success, robot_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=ROBOT_PRIM_PATH,
    )
    if not success:
        raise RuntimeError(f"Failed to import URDF: {{urdf_path}}")
    return robot_path


class {class_name}Env:
    """Minimal Gym-style RL environment stub wrapping the imported
    '{robot_name}' articulation. `reset()`/`step()` are intentionally
    empty of any task logic (reward shaping, termination, domain
    randomization) — that's this environment's real design work, specific
    to whatever task '{assembly_name}' is being trained for, and can't be
    guessed from the URDF alone."""

    def __init__(self) -> None:
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        robot_prim_path = _import_urdf_as_usd(URDF_PATH)
        add_reference_to_stage(usd_path=URDF_PATH, prim_path=robot_prim_path)

        self.robot = Articulation(prim_path=robot_prim_path, name=ROBOT_NAME)
        self.world.scene.add(self.robot)
        self.world.reset()

    def reset(self) -> torch.Tensor:
        """TODO: randomize initial joint state / targets for your task.
        Returns the initial observation tensor."""
        self.world.reset()
        return self._get_observation()

    def step(self, action: torch.Tensor):
        """TODO: convert `action` into a real ArticulationAction, compute
        a real reward/done/info for your task. `action` should have
        `self.robot.num_dof` entries. Currently a no-op physics tick."""
        self.world.step(render=True)
        observation = self._get_observation()
        reward = torch.zeros(1)  # TODO
        done = torch.zeros(1, dtype=torch.bool)  # TODO
        info: dict = {{}}
        return observation, reward, done, info

    def _get_observation(self) -> torch.Tensor:
        positions = self.robot.get_joint_positions()
        velocities = self.robot.get_joint_velocities()
        return torch.as_tensor(np.concatenate([positions, velocities]), dtype=torch.float32)

    def close(self) -> None:
        simulation_app.close()


def main() -> None:
    env = {class_name}Env()
    try:
        env.reset()
        zero_action = torch.zeros(env.robot.num_dof)
        for _ in range(200):
            env.step(zero_action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
'''


def _ros2_launch_script(
    assembly_name: str, robot_name: str, urdf_path: Path, output_filename: str, *, include_gazebo: bool
) -> str:
    """A plain ROS2 Python launch file (``launch``/``launch_ros`` — the
    standard, ``xacro``-free "just read the .urdf file's own text" launch
    pattern, since '{urdf_path.name}' is already a plain ``.urdf``, not a
    ``.xacro`` template needing preprocessing) that starts
    ``robot_state_publisher`` (publishes ``/robot_description`` + TF from
    the URDF) plus, for ``target_platform="gazebo"`` only,
    ``gazebo_ros``'s own ``gazebo.launch.py`` and ``spawn_entity.py`` to
    actually place the robot into a running Gazebo world — the plain
    ``"ros2"`` target starts only ``robot_state_publisher``/
    ``joint_state_publisher`` (e.g. for RViz2 visualization or a
    ``ros2_control``-driven simulator with no Gazebo dependency at all).
    """
    # repr(), not manual "..."/r"..." wrapping — see _isaac_sim_script's own
    # comment for why (a literal `"`/`\` in robot_name/urdf_path would
    # otherwise break the URDF_PATH/ROBOT_NAME assignment below).
    urdf_path_repr = repr(str(urdf_path))
    robot_name_repr = repr(robot_name)
    header = f'''"""Auto-generated ROS2 launch file for '{assembly_name}' — produced by
Dana's generate_simulation_wrapper tool from {urdf_path.name}.

Run with:
    ros2 launch {output_filename}
(from a sourced ROS2 workspace — {'gazebo_ros, ' if include_gazebo else ''}robot_state_publisher{
        '' if include_gazebo else ', joint_state_publisher'
    } must be installed).
"""

from __future__ import annotations
'''
    if include_gazebo:
        body = f'''import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

URDF_PATH = {urdf_path_repr}
ROBOT_NAME = {robot_name_repr}


def generate_launch_description() -> LaunchDescription:
    with open(URDF_PATH, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        )
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{{"robot_description": robot_description, "use_sim_time": True}}],
    )

    # Spawns ROBOT_NAME into the already-running Gazebo world from the
    # /robot_description topic robot_state_publisher above just published —
    # the standard gazebo_ros bridge, no SDF conversion step needed since
    # gazebo_ros's own <gazebo> URDF plugin support reads plain URDF+xacro
    # tags directly.
    spawn_entity_node = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", ROBOT_NAME],
        output="screen",
    )

    return LaunchDescription(
        [
            gazebo,
            robot_state_publisher_node,
            spawn_entity_node,
        ]
    )
'''
    else:
        body = f'''from launch import LaunchDescription
from launch_ros.actions import Node

URDF_PATH = {urdf_path_repr}
ROBOT_NAME = {robot_name_repr}


def generate_launch_description() -> LaunchDescription:
    with open(URDF_PATH, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    # No Gazebo dependency: this publishes /robot_description + TF only —
    # enough to visualize '{robot_name}' in RViz2 or drive it from a
    # separately-launched ros2_control/custom simulator. Use
    # target_platform="gazebo" instead for a full physics simulation.
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{{"robot_description": robot_description}}],
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{{"robot_description": robot_description}}],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            joint_state_publisher_node,
        ]
    )
'''
    return header + "\n" + body


def _webots_controller_script(assembly_name: str, robot_name: str, urdf_path: Path, output_filename: str) -> str:
    """Webots robot controller: unlike the Isaac Sim/ROS2 targets above,
    Webots has no single runtime call that loads an arbitrary ``.urdf``
    file straight into a running simulation — a URDF must first be
    converted OFFLINE into a Webots PROTO node via the ``urdf2webots``
    converter (``pip install urdf2webots``; Cyberbotics' own official
    converter, not a Dana-specific tool), and that PROTO then placed in
    the ``.wbt`` world file with this generated script set as its
    ``controller`` field — the idiomatic Webots workflow (a controller
    drives an ALREADY-PLACED robot, it doesn't import its own model the
    way ``omni.kit.commands`` or a ROS2 launch file can). This script is
    therefore that robot's controller, not a standalone loader: it
    enumerates every ``Motor``-type device on the robot generically via
    ``Robot.getNumberOfDevices``/``getDeviceByIndex`` (real, version-
    stable Webots controller API — no hardcoded per-joint device names,
    since those come from ``urdf2webots``'s own naming and aren't known
    ahead of time here), then wraps them in the same minimal Gym-style
    env shape (``reset()``/``step()`` stubs) the Isaac Sim target uses,
    for a consistent shape across every target this module generates.
    """
    class_name = "".join(w[0].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", assembly_name) if w) or "Robot"
    # repr(), not manual "..."/r"..." wrapping — see _isaac_sim_script's own
    # comment for why.
    urdf_path_repr = repr(str(urdf_path))
    robot_name_repr = repr(robot_name)
    proto_name = re.sub(r"[^A-Za-z0-9_]", "_", robot_name).strip("_") or "Robot"
    return f'''"""Auto-generated Webots controller for '{assembly_name}' — produced by
Dana's generate_simulation_wrapper tool from {urdf_path.name}.

Webots has no single call that loads a raw .urdf at runtime — convert it
to a PROTO OFFLINE first, then place that PROTO in your .wbt world file
with THIS script set as its controller:

    pip install urdf2webots
    python -m urdf2webots.importer --input={urdf_path_repr} --output={proto_name}.proto

(the urdf2webots CLI's own flags have changed across releases -- see
https://github.com/cyberbotics/urdf2webots for the version you installed)
then in Webots: Add Node > your {proto_name} PROTO, set its `controller`
field to "{Path(output_filename).stem}".

Run from inside Webots itself (this is a robot controller, not a
standalone script) — Webots launches it automatically once the above
`controller` field points at it.
"""

from __future__ import annotations

from controller import Node, Supervisor

ROBOT_NAME = {robot_name_repr}
# Real, version-stable Webots node-type constants for the two motor
# kinds a converted URDF's revolute/prismatic joints become.
_MOTOR_NODE_TYPES = (Node.ROTATIONAL_MOTOR, Node.LINEAR_MOTOR)


def _discover_motors(robot: Supervisor) -> list:
    """Every Motor-type device on this robot, found generically (no
    hardcoded per-joint names — see this module's own docstring for why)
    via Robot.getNumberOfDevices/getDeviceByIndex, both real Webots
    controller API since Webots R2023a."""
    motors = []
    for i in range(robot.getNumberOfDevices()):
        device = robot.getDeviceByIndex(i)
        if device.getNodeType() in _MOTOR_NODE_TYPES:
            motors.append(device)
    return motors


class {class_name}Env:
    """Minimal Gym-style RL environment stub wrapping '{robot_name}' as a
    Webots Supervisor. `reset()`/`step()` are intentionally empty of any
    task logic (reward shaping, termination, domain randomization) —
    that's this environment's real design work, specific to whatever
    task '{assembly_name}' is being trained for, and can't be guessed
    from the URDF alone."""

    def __init__(self) -> None:
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.motors = _discover_motors(self.robot)
        for motor in self.motors:
            motor.setPosition(float("inf"))  # velocity control mode
            motor.setVelocity(0.0)

    def reset(self):
        """TODO: randomize initial joint state / targets for your task."""
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        return self._get_observation()

    def step(self, action):
        """TODO: apply `action` to self.motors, compute a real
        reward/done/info for your task. Currently a no-op physics tick."""
        self.robot.step(self.timestep)
        observation = self._get_observation()
        reward = 0.0  # TODO
        done = False  # TODO
        info: dict = {{}}
        return observation, reward, done, info

    def _get_observation(self):
        return [m.getTargetPosition() for m in self.motors]

    def close(self) -> None:
        pass  # Webots owns the process lifecycle; nothing to tear down here.


def main() -> None:
    env = {class_name}Env()
    env.reset()
    zero_action = [0.0] * len(env.motors)
    for _ in range(200):
        env.step(zero_action)


if __name__ == "__main__":
    main()
'''


def generate_simulation_wrapper(
    assembly_name: str,
    target_platform: str,
    urdf_path: str,
    output_path: str | None = None,
) -> str:
    """Writes a standalone Python simulation-loader script for
    ``assembly_name``'s already-exported ``urdf_path`` — the last mile
    between ``export_assembly_to_urdf`` and an actual RL training run.

    ``target_platform``:

    - ``"isaac_sim"``: a standalone Isaac Sim script (boots
      ``SimulationApp``, imports the URDF via the URDF importer
      extension, spawns it at the origin on a ground plane) wrapped in a
      minimal Gym-style env class with empty ``reset()``/``step()``
      stubs operating on PyTorch tensors — see ``_isaac_sim_script``'s
      own docstring for exactly what's stubbed and why.
    - ``"gazebo"``: a ROS2 Python launch file that starts
      ``robot_state_publisher`` + Gazebo (``gazebo_ros``'s own
      ``gazebo.launch.py``) and spawns the robot into it via
      ``spawn_entity.py``.
    - ``"ros2"``: the same launch file shape without any Gazebo
      dependency — just ``robot_state_publisher``/``joint_state_publisher``
      (e.g. for RViz2 visualization or a ``ros2_control``-driven
      simulator).
    - ``"webots"``: a Webots robot controller script (``controller``
      module's ``Supervisor`` API) — NOT a standalone loader, since
      Webots has no runtime call that ingests a raw ``.urdf`` directly;
      see ``_webots_controller_script``'s own docstring for the required
      offline ``urdf2webots`` conversion step and why this script's shape
      differs from the other two targets.

    ``urdf_path`` is resolved against a few likely locations before it's
    ever embedded into the generated script (see ``_resolve_urdf_path``)
    so a hallucinated path fails loudly here instead of producing a
    script that immediately errors when run. ``output_path`` defaults to
    a ``<assembly_name>_<target_platform>_sim.py`` file under this
    session's own ``freecad_output/sim_wrappers/`` directory (Workspace
    Hygiene — see ``_SIM_WRAPPERS_SUBDIR``'s own comment for why this
    isn't a sibling of the resolved ``.urdf`` the way it used to be).
    """
    assembly = (assembly_name or "").strip()
    if not assembly:
        return _error("generate_simulation_wrapper requires assembly_name")
    platform = (target_platform or "").strip().lower()
    if platform not in _TARGET_PLATFORMS:
        return _error(
            f"generate_simulation_wrapper: unknown target_platform {platform!r} — "
            f"must be one of {sorted(_TARGET_PLATFORMS)}"
        )
    raw_urdf_path = (urdf_path or "").strip()
    if not raw_urdf_path:
        return _error("generate_simulation_wrapper requires urdf_path")

    resolved_urdf = _resolve_urdf_path(raw_urdf_path)
    if resolved_urdf is None:
        return _error(
            f"generate_simulation_wrapper: urdf_path {raw_urdf_path!r} does not exist. You MUST use "
            "'export_assembly_to_urdf' or 'generate_urdf_assembly' to produce a real .urdf file first."
        )

    safe = _safe_name(assembly)
    robot_name = _robot_name_from_urdf(resolved_urdf, fallback=safe)

    out_path = Path(output_path) if (output_path or "").strip() else (
        session_scoped_dir(_FREECAD_OUTPUT_DIR) / _SIM_WRAPPERS_SUBDIR / f"{safe}_{platform}_sim.py"
    )

    if platform == "isaac_sim":
        script = _isaac_sim_script(assembly, robot_name, resolved_urdf, out_path.name)
    elif platform == "webots":
        script = _webots_controller_script(assembly, robot_name, resolved_urdf, out_path.name)
    else:
        script = _ros2_launch_script(
            assembly, robot_name, resolved_urdf, out_path.name, include_gazebo=(platform == "gazebo")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")

    if platform == "isaac_sim":
        instructions = (
            f"Run with Isaac Sim's own bundled Python interpreter (not a system Python): "
            f"<isaac-sim-install-dir>/python.sh {out_path} (Linux) or "
            f"<isaac-sim-install-dir>/python.bat {out_path} (Windows). Requires a local Isaac Sim "
            "installation; the omni.importer.urdf import at the top may need adjusting for your version "
            "(see the script's own header comment)."
        )
    elif platform == "webots":
        instructions = (
            f"Not a standalone script -- convert {resolved_urdf.name} to a PROTO first with "
            f"'pip install urdf2webots && python -m urdf2webots.importer', add that PROTO to your .wbt "
            f"world file, and set its controller field to {out_path.stem!r}. Requires a local Webots "
            "installation (see the script's own header comment for the exact conversion command)."
        )
    else:
        instructions = (
            f"Launch with: ros2 launch {out_path} (from a sourced ROS2 workspace). Requires "
            + ("gazebo_ros, " if platform == "gazebo" else "")
            + "robot_state_publisher"
            + ("" if platform == "gazebo" else ", joint_state_publisher")
            + " installed."
        )

    return _ok(
        name=safe,
        type="sim_wrapper",
        target_platform=platform,
        path=str(out_path),
        urdf_path=str(resolved_urdf),
        robot_name=robot_name,
        message=f"Wrote {platform} simulation wrapper for '{assembly}' to {out_path}. {instructions}",
        instructions=instructions,
    )
