"""Image-to-3D mesh generation — outsourced to a cloud service rather than
running any reconstruction model locally, so this stays a thin, lightweight
client rather than a new local GPU dependency.

Two tiers, tried in order, each a normal fallback into the next:

1. **Dedicated REST APIs** (Tripo3D, Meshy) — paid, reliable, no cold-start
   queueing. Opt-in via ``TRIPO_API_KEY``/``MESHY_API_KEY`` env vars; skipped
   entirely when neither is set. Implemented with plain ``requests`` (already
   a project dependency — see ``dana.core.model_provider``), not a vendor
   SDK, so this adds zero new packages/heaviness.
2. **Free Hugging Face Spaces** (unchanged) — polls a fixed chain of public
   Spaces via ``gradio_client`` (the correct way to drive a Space's actual
   Gradio API — replicating its queueing/upload/download protocol over plain
   ``requests`` would mean re-implementing a chunk of Gradio's own client
   internals for no real benefit) until one returns a usable ``.obj``/
   ``.glb`` mesh file, or all of them have failed. This is the path every
   caller falls back to when no dedicated API key is configured, or when the
   configured one errors out.

Free community Spaces routinely sleep, queue for minutes on a cold start,
change their API shape without notice, or simply go down — this fallback
chain, and the broad per-attempt exception handling below, are the actual
point of tier 2, not an afterthought. ``Zhengyi/CRM`` (attempt 2) was
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

import base64
import json
import mimetypes
import os
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Callable

import requests

from dana.paths import DANA_WORKSPACE
from dana.session_context import session_scoped_dir

# Same base directory dana.plugins.freecad.engine._OUTPUT_DIR writes
# create_freecad_*/export_mesh_stl artifacts to — declared as its own copy
# rather than importing engine.py's underscore-prefixed module attribute
# across modules, same precedent dana.api.cad/dana.tools.urdf_builder's own
# docstrings already apply. ``_session_dir()`` (not the bare constant) is
# what the actual download destination below is built from, so a mesh this
# session generates never lands in a directory another session's own
# generate_3d_from_image call could also write into.
_OUTPUT_DIR = DANA_WORKSPACE / "freecad_output"


def _session_dir() -> Path:
    return session_scoped_dir(_OUTPUT_DIR)

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

# Same overall budget as one Gradio backend attempt above, applied to one
# dedicated-API attempt (upload + create-task + the full poll loop). A paid
# API shouldn't ordinarily need anywhere near this long, but the wall-clock
# cap still guards against a hung request/an endpoint that never reaches a
# terminal status.
_DEDICATED_API_TIMEOUT_S = 180.0
_DEDICATED_API_POLL_INTERVAL_S = 3.0

_TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
_MESHY_BASE_URL = "https://api.meshy.ai"


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


def _guess_image_mime(image: Path) -> tuple[str, str]:
    """Returns ``(mime_type, short_extension)``, e.g. ``("image/png",
    "png")`` — both APIs below need one or the other (a MIME type for the
    upload/data-URI, a bare extension for Tripo3D's ``file.type`` field).
    Defaults to PNG for anything mimetypes doesn't recognize (e.g. no
    extension at all) rather than failing the call outright."""
    mime, _ = mimetypes.guess_type(image.name)
    mime = mime or "image/png"
    return mime, mime.rsplit("/", maxsplit=1)[-1]


def _download_mesh(url: str, suffix: str) -> str:
    """Downloads a dedicated API's (time-limited, pre-signed) result URL to
    a local temp file — same shape ``gradio_client`` itself already hands
    back to ``_run_triposr``/etc., so ``_is_valid_mesh_file``/
    ``_move_to_output`` treat either tier's output identically."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(response.content)
    return tmp_path


def _run_tripo3d(image_path: str, api_key: str) -> str | None:
    """Tripo3D's official REST API (https://docs.tripo3d.ai) — upload the
    image, create an ``image_to_model`` task, poll it to a terminal status,
    download the resulting model. A paid, reliable alternative to the free
    Gradio Spaces below; only ever called when ``TRIPO_API_KEY`` is set.

    Any shape this doesn't expect (an unrecognized status, a missing output
    URL) is treated as "no mesh" rather than raised — the caller's own
    broad ``except Exception`` around this whole call already falls through
    to the next dedicated API / the Gradio chain for any raised error, but a
    clean ``None`` return covers the "call succeeded, result just wasn't
    usable" case the same way.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    image = Path(image_path)
    mime, ext = _guess_image_mime(image)

    with open(image, "rb") as fh:
        upload = requests.post(
            f"{_TRIPO_BASE_URL}/upload",
            headers=headers,
            files={"file": (image.name, fh, mime)},
            timeout=60,
        )
    upload.raise_for_status()
    image_token = upload.json()["data"]["image_token"]

    task = requests.post(
        f"{_TRIPO_BASE_URL}/task",
        headers=headers,
        json={"type": "image_to_model", "file": {"type": ext, "file_token": image_token}},
        timeout=30,
    )
    task.raise_for_status()
    task_id = task.json()["data"]["task_id"]

    deadline = time.monotonic() + _DEDICATED_API_TIMEOUT_S
    while time.monotonic() < deadline:
        poll = requests.get(f"{_TRIPO_BASE_URL}/task/{task_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json().get("data") or {}
        status = data.get("status")
        if status == "success":
            output = data.get("output") or {}
            model_url = output.get("pbr_model") or output.get("model") or output.get("base_model")
            return _download_mesh(model_url, ".glb") if model_url else None
        if status in {"failed", "banned", "expired", "cancelled", "unknown"}:
            return None
        time.sleep(_DEDICATED_API_POLL_INTERVAL_S)
    return None


def _run_meshy(image_path: str, api_key: str) -> str | None:
    """Meshy's official REST API (https://docs.meshy.ai) — POST the image as
    a base64 data URI (no separate upload step, unlike Tripo3D above), poll
    the task to a terminal status, download the resulting glb. A paid,
    reliable alternative to the free Gradio Spaces below; only ever called
    when ``MESHY_API_KEY`` is set. Same "unexpected shape -> None, not
    raised" reasoning as ``_run_tripo3d`` above.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    image = Path(image_path)
    mime, _ext = _guess_image_mime(image)
    data_uri = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"

    create = requests.post(
        f"{_MESHY_BASE_URL}/openapi/v1/image-to-3d",
        headers=headers,
        json={"image_url": data_uri, "enable_pbr": True},
        timeout=60,
    )
    create.raise_for_status()
    task_id = create.json()["result"]

    deadline = time.monotonic() + _DEDICATED_API_TIMEOUT_S
    while time.monotonic() < deadline:
        poll = requests.get(f"{_MESHY_BASE_URL}/openapi/v1/image-to-3d/{task_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        if status == "SUCCEEDED":
            model_url = (data.get("model_urls") or {}).get("glb")
            return _download_mesh(model_url, ".glb") if model_url else None
        if status in {"FAILED", "CANCELED"}:
            return None
        time.sleep(_DEDICATED_API_POLL_INTERVAL_S)
    return None


# Checked in this order — first env var present wins. Each entry's run_fn
# takes (image_path, api_key) and returns a local mesh path or None,
# matching _BACKENDS' (Any, str) -> str | None shape closely enough to
# share _call_with_timeout/_is_valid_mesh_file/_move_to_output below.
_DEDICATED_APIS: tuple[tuple[str, str, Callable[[str, str], str | None]], ...] = (
    ("TRIPO_API_KEY", "Tripo3D", _run_tripo3d),
    ("MESHY_API_KEY", "Meshy", _run_meshy),
)


def _try_dedicated_apis(image: Path) -> tuple[str, str] | None:
    """Tier 1 of ``generate_3d_from_image``: for the first of
    ``_DEDICATED_APIS`` whose env var is actually set, calls that service's
    official REST API instead of the free Gradio Spaces in tier 2 — a paid
    endpoint doesn't suffer their cold-start timeouts/undocumented API
    drift. Returns ``None`` (never raises) when no key is configured, or
    when the configured service's call times out, raises, or comes back
    without a usable mesh — either way the caller falls straight through to
    the Gradio chain, exactly as if this tier didn't exist.
    """
    for env_var, label, run_fn in _DEDICATED_APIS:
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            continue
        try:
            local_path = _call_with_timeout(
                lambda run_fn=run_fn, api_key=api_key: run_fn(str(image), api_key),
                timeout=_DEDICATED_API_TIMEOUT_S,
            )
        except Exception:  # noqa: BLE001 — a dedicated-API failure just falls through to the Gradio chain
            continue
        if _is_valid_mesh_file(local_path):
            return local_path, label
    return None


# Polling order the Gradio fallback chain follows, exactly as specified:
# TripoSR first, then CRM, then zero123plus. "Zhengyi/CRM" (not the
# "Zhenanyi/CRM" spelling sometimes seen) is the actual Space id.
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
    """Moves the backend's downloaded temp file into THIS session's own
    output directory — a random suffix keeps two calls to the SAME backend
    in one session from colliding on identical filenames."""
    src = Path(local_path)
    dest = _session_dir() / f"{_safe_label(model_label)}_{uuid.uuid4().hex[:8]}{src.suffix.lower()}"
    shutil.move(str(src), dest)
    return dest


def _call_with_timeout(fn: Callable[[], str | None], *, timeout: float) -> str | None:
    """Runs ``fn`` (a blocking ``gradio_client`` call, or a dedicated-API
    upload+poll call) in a worker thread and bounds how long it's allowed to
    block — neither ``gradio_client`` nor the ``requests``-based dedicated
    clients above have a single built-in call-level timeout of their own,
    and a Space cold-starting/queued indefinitely (or an API stuck between
    polls) would otherwise hang this whole tool call rather than falling
    through to the next backend.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        return future.result(timeout=timeout)


def generate_3d_from_image(image_path: str) -> str:
    """Convert a 2D image into a 3D mesh.

    Tries ``TRIPO_API_KEY``/``MESHY_API_KEY``'s dedicated REST API first (in
    that order — see ``_try_dedicated_apis``), then falls back to polling
    free Hugging Face Spaces in order (TripoSR, then CRM, then zero123plus)
    if no key is set or the dedicated call didn't produce a usable mesh —
    moving whichever attempt succeeds first into ``freecad_output/`` and
    returning its path.

    Returns ``{"ok": true, "mesh_path": str, "model_used": str}`` on
    success, or ``{"ok": false, "error": "All Image-to-3D endpoints timed
    out or failed. Please try again later."}`` if every attempt fails.
    """
    image = Path(image_path)
    if not image.is_file():
        return _error(f"generate_3d_from_image: image_path not found: {image_path}")

    dedicated = _try_dedicated_apis(image)
    if dedicated is not None:
        local_path, label = dedicated
        try:
            dest = _move_to_output(local_path, label)
            return _ok(mesh_path=str(dest), model_used=label)
        except OSError:
            pass  # fall through to the Gradio chain below

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
