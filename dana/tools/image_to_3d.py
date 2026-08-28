"""Image-to-3D mesh generation — outsourced to free Hugging Face Spaces
rather than running any reconstruction model locally, so this stays a thin,
lightweight client rather than a new local GPU dependency.

Polls a fixed chain of public Spaces via ``gradio_client`` (the correct way
to drive a Space's actual Gradio API — replicating its queueing/upload/
download protocol over plain ``requests`` would mean re-implementing a
chunk of Gradio's own client internals for no real benefit) until one
returns a usable ``.obj``/``.glb`` mesh file, or all of them have failed.

Free community Spaces routinely sleep, queue for minutes on a cold start,
change their API shape without notice, or simply go down — this fallback
chain, and the broad per-attempt exception handling below, are the actual
point of this module, not an afterthought. ``Zhengyi/CRM`` (attempt 2) was
returning a runtime error and ``sudo-ai/zero123plus-v1.1`` (attempt 3) was
failing to schedule at all as of this writing — an expected, ordinary state
for this class of Space, which is exactly why there are 3 attempts and not 1.

Note on attempt 3: ``zero123plus`` is a novel-view-synthesis diffusion
model, not a mesh reconstructor — its real output is a grid of generated
2D views, not a ``.obj``/``.glb`` mesh. It's still polled (the fallback
chain names it explicitly), but ``_is_valid_mesh_file``'s extension check
means a non-mesh response correctly fails this attempt — falling through
to the final "all endpoints failed" error — rather than being mislabeled
as a successful mesh.

Every public function here returns a JSON string (``{"ok": bool, ...}``),
matching every other ``dana.tools``/``dana.plugins.freecad`` tool's wire
contract (see ``dana.plugins.freecad.engine``'s own module docstring) —
not a plain ``dict`` — so it slots into ``dana.core.react_dispatch``'s
handler dict (via a ``json.loads``-wrapping ``_tool_*`` handler, same
pattern as ``generate_urdf_assembly``) the same way.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable

from dana.paths import DANA_WORKSPACE

# Same directory dana.plugins.freecad.engine._OUTPUT_DIR writes
# create_freecad_*/export_mesh_stl artifacts to — declared as its own copy
# rather than importing engine.py's underscore-prefixed module attribute
# across modules, same precedent dana.api.cad/dana.tools.urdf_builder's own
# docstrings already apply.
_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"

_MESH_EXTENSIONS = (".glb", ".obj")

# A real mesh file is never this small — guards against a Space returning an
# empty/placeholder/error-page file that still happens to have a .obj/.glb
# name, so that never gets mistaken for a valid result.
_MIN_MESH_BYTES = 64

# Free Spaces can queue for minutes waking from a cold sleep; this bounds
# how long ONE attempt is allowed to block before the chain gives up on it
# and moves to the next one, since gradio_client.Client.predict() has no
# built-in call-level timeout of its own.
_BACKEND_TIMEOUT_S = 180.0


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": str(message)})


def _first_mesh_path(result: Any) -> str | None:
    """``gradio_client.Client.predict()`` already downloads any File/Model3D
    output component to a local temp path — this flattens a (possibly
    nested) tuple/list result and picks the first string that looks like a
    local path, preferring a ``.glb`` (one self-contained binary, no
    companion ``.mtl``/texture files a viewer would also need) over a
    ``.obj`` when both are returned — TripoSR's own ``generate`` returns
    exactly that pair.
    """
    stack = list(result) if isinstance(result, (list, tuple)) else [result]
    flat: list[Any] = []
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack[:0] = list(item)
        else:
            flat.append(item)
    paths = [str(item) for item in flat if isinstance(item, str) and item]
    glb = next((p for p in paths if p.lower().endswith(".glb")), None)
    return glb or (paths[0] if paths else None)


def _run_triposr(client: Any, image_path: str) -> str | None:
    """TripoSR's own demo UI chains ``preprocess`` -> ``generate`` (two
    separate Gradio events, not one combined predict endpoint) — confirmed
    against the live Space's ``app.py`` Blocks wiring: neither sets an
    explicit ``api_name``, so each is auto-exposed at its own
    function-name route.

    ``do_remove_background=True``/``foreground_ratio=0.85``/
    ``mc_resolution=256`` mirror that app.py's own UI defaults.
    """
    processed = client.predict(image_path, True, 0.85, api_name="/preprocess")
    result = client.predict(processed, 256, api_name="/generate")
    return _first_mesh_path(result)


def _run_crm(client: Any, image_path: str) -> str | None:
    """``Zhengyi/CRM``'s exact API surface isn't independently confirmed
    here (the Space was returning a runtime error, not a live demo, as of
    this writing) — one generic ``predict`` call, so either a shape
    mismatch or an outright-down Space just fails this attempt and falls
    through to the next backend, same as any other incompatible endpoint.
    """
    result = client.predict(image_path, api_name="/predict")
    return _first_mesh_path(result)


def _run_zero123plus(client: Any, image_path: str) -> str | None:
    """See this module's docstring note on attempt 3 — real output here is
    multi-view images, not a mesh, so this is expected to fail
    ``_is_valid_mesh_file`` even when the call itself succeeds."""
    result = client.predict(image_path, api_name="/predict")
    return _first_mesh_path(result)


# Polling order the fallback chain follows, exactly as specified: TripoSR
# first, then CRM, then zero123plus. "Zhengyi/CRM" (not the "Zhenanyi/CRM"
# spelling sometimes seen) is the actual Space id.
_BACKENDS: tuple[tuple[str, str, Callable[[Any, str], str | None]], ...] = (
    ("stabilityai/TripoSR", "TripoSR", _run_triposr),
    ("Zhengyi/CRM", "CRM", _run_crm),
    ("sudo-ai/zero123plus-v1.1", "zero123plus-v1.1", _run_zero123plus),
)


def _is_valid_mesh_file(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.is_file() and p.suffix.lower() in _MESH_EXTENSIONS and p.stat().st_size >= _MIN_MESH_BYTES


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", label or "").strip("_") or "mesh"


def _move_to_output(local_path: str, model_label: str) -> Path:
    """Moves the backend's downloaded temp file into the canonical
    ``freecad_output/`` — a random suffix keeps two calls to the SAME
    backend in one session from colliding on identical filenames."""
    src = Path(local_path)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = _OUTPUT_DIR / f"{_safe_label(model_label)}_{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
    shutil.move(str(src), dest)
    return dest


def _call_with_timeout(fn: Callable[[], str | None], *, timeout: float) -> str | None:
    """Runs ``fn`` (a blocking ``gradio_client`` call) in a worker thread and
    bounds how long it's allowed to block — ``gradio_client`` itself has no
    built-in per-call timeout, and a Space cold-starting/queued indefinitely
    would otherwise hang this whole tool call rather than falling through
    to the next backend.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        return future.result(timeout=timeout)


def generate_3d_from_image(image_path: str) -> str:
    """Convert a 2D image into a 3D mesh by polling free Hugging Face
    Spaces in order (TripoSR, then CRM, then zero123plus), moving whichever
    one succeeds first into ``freecad_output/`` and returning its path.

    Returns ``{"ok": true, "mesh_path": str, "model_used": str}`` on
    success, or ``{"ok": false, "error": "All Image-to-3D endpoints timed
    out or failed. Please try again later."}`` if every backend fails.
    """
    image = Path(image_path)
    if not image.is_file():
        return _error(f"generate_3d_from_image: image_path not found: {image_path}")

    try:
        from gradio_client import Client
    except ImportError:
        return _error("generate_3d_from_image: the 'gradio_client' package is not installed")

    for space_id, label, run_fn in _BACKENDS:
        try:
            client = Client(space_id)
            local_path = _call_with_timeout(
                lambda: run_fn(client, str(image)), timeout=_BACKEND_TIMEOUT_S
            )
        except FutureTimeoutError:
            continue
        except Exception:  # noqa: BLE001 — any failure here just advances to the next fallback
            continue
        if not _is_valid_mesh_file(local_path):
            continue
        try:
            dest = _move_to_output(local_path, label)
        except OSError:
            continue
        return _ok(mesh_path=str(dest), model_used=label)

    return _error("All Image-to-3D endpoints timed out or failed. Please try again later.")
