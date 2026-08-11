"""AutoCAD COM automation operator — deterministic drawing via win32com.client.

Guardrail: AutoCAD geometry is NEVER driven by mouse clicks or pixel
coordinates (that would be non-deterministic and DPI/zoom-fragile). Every
primitive here goes straight to the ObjectARX model-space COM API or the
command line via ``SendCommand``/AutoLISP, so results are exact regardless
of window position, zoom level, or screen resolution.

All public functions return a JSON string (``{"ok": bool, ...}``) so they
drop straight into the tool broker's string-observation contract — see
``dana.tools.broker.initialize_tool_registry`` for the tool_id wiring.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Sequence

import pythoncom
import win32com.client

from dana.security.dry_run import is_dry_run_enabled

_ACAD_PROGID = "AutoCAD.Application"

# AutoCAD's COM automation object is not documented as thread-safe for
# concurrent calls — serialize every primitive the same way the physical
# desktop actuators do (single foreground owner, dana.middleware.actuator_executor).
_lock = threading.Lock()


class AutoCADConnectionError(RuntimeError):
    """Raised when AutoCAD isn't running, has no active document, or COM dispatch fails."""


def _ok(**payload: Any) -> str:
    return json.dumps({"ok": True, **payload})


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": str(message)})


def _dry_run_result(op: str, **payload: Any) -> str:
    return _ok(op=op, dry_run=True, **payload)


def _variant_point(pt: Sequence[float]) -> Any:
    """AutoCAD COM points are a SAFEARRAY of 3 doubles (z defaults to 0)."""
    coords = list(pt)
    if len(coords) < 3:
        coords = coords + [0.0] * (3 - len(coords))
    return win32com.client.VARIANT(
        pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(c) for c in coords[:3]]
    )


def _variant_points(points: Sequence[Sequence[float]]) -> Any:
    """Flattened (x, y) SAFEARRAY of doubles for AddLightWeightPolyline."""
    flat: list[float] = []
    for pt in points:
        coords = list(pt)
        if len(coords) < 2:
            raise ValueError(f"polyline point needs at least (x, y): {pt!r}")
        flat.extend([float(coords[0]), float(coords[1])])
    return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, flat)


def _variant_object_array(objects: Sequence[Any]) -> Any:
    """SAFEARRAY of IDispatch objects, as ``AddRegion`` expects."""
    return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, list(objects))


def get_application(*, start_if_missing: bool = True) -> Any:
    """Return the live ``AutoCAD.Application`` COM object.

    Tries ``GetActiveObject`` first — attach to an already-running AutoCAD,
    the expected case since a human or a prior tool call opened it — and
    only falls back to ``Dispatch`` (launch a new instance) when
    ``start_if_missing`` is set. Raises ``AutoCADConnectionError`` on any
    COM failure rather than letting a raw ``pywintypes.com_error`` leak
    into the tool broker.
    """
    pythoncom.CoInitialize()
    try:
        app = win32com.client.GetActiveObject(_ACAD_PROGID)
    except Exception as exc:  # noqa: BLE001
        if not start_if_missing:
            raise AutoCADConnectionError(
                f"AutoCAD is not running (GetActiveObject failed: {exc})"
            ) from exc
        try:
            app = win32com.client.Dispatch(_ACAD_PROGID)
        except Exception as dispatch_exc:  # noqa: BLE001
            raise AutoCADConnectionError(
                f"AutoCAD COM dispatch failed: {dispatch_exc}"
            ) from dispatch_exc
    try:
        app.Visible = True
    except Exception as exc:  # noqa: BLE001
        raise AutoCADConnectionError(f"AutoCAD COM object is not usable: {exc}") from exc
    return app


def get_active_document(app: Any = None) -> Any:
    app = app if app is not None else get_application()
    try:
        return app.ActiveDocument
    except Exception as exc:  # noqa: BLE001
        raise AutoCADConnectionError(f"no active AutoCAD document: {exc}") from exc


def _model_space(doc: Any = None) -> Any:
    doc = doc if doc is not None else get_active_document()
    return doc.ModelSpace


def add_line(start_pt: Sequence[float], end_pt: Sequence[float]) -> str:
    """Draw a line from ``start_pt`` to ``end_pt`` (model-space ``[x, y]`` or ``[x, y, z]``)."""
    if is_dry_run_enabled():
        return _dry_run_result("add_line", start=list(start_pt), end=list(end_pt))
    with _lock:
        try:
            ms = _model_space()
            obj = ms.AddLine(_variant_point(start_pt), _variant_point(end_pt))
            return _ok(
                op="add_line", handle=str(obj.Handle), start=list(start_pt), end=list(end_pt)
            )
        except AutoCADConnectionError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(f"add_line failed: {exc}")


def add_circle(center_pt: Sequence[float], radius: float) -> str:
    """Draw a circle centered at ``center_pt`` with the given ``radius``."""
    if is_dry_run_enabled():
        return _dry_run_result("add_circle", center=list(center_pt), radius=radius)
    with _lock:
        try:
            ms = _model_space()
            obj = ms.AddCircle(_variant_point(center_pt), float(radius))
            return _ok(
                op="add_circle", handle=str(obj.Handle), center=list(center_pt), radius=radius
            )
        except AutoCADConnectionError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(f"add_circle failed: {exc}")


def add_polyline(points_list: Sequence[Sequence[float]]) -> str:
    """Draw a lightweight polyline through ``points_list`` (each a 2D ``[x, y]``)."""
    if len(points_list) < 2:
        return _error("add_polyline requires at least 2 points")
    if is_dry_run_enabled():
        return _dry_run_result("add_polyline", points=[list(p) for p in points_list])
    with _lock:
        try:
            ms = _model_space()
            obj = ms.AddLightWeightPolyline(_variant_points(points_list))
            return _ok(
                op="add_polyline",
                handle=str(obj.Handle),
                points=[list(p) for p in points_list],
            )
        except AutoCADConnectionError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(f"add_polyline failed: {exc}")


def add_extruded_solid(profile_polyline: Sequence[Sequence[float]], height: float) -> str:
    """Extrude a closed polyline profile into a 3D solid.

    Draws the profile as a closed lightweight polyline, converts it to a
    region (``AddRegion``), then extrudes that region ``height`` units via
    ``AddExtrudedSolid`` — the standard COM path for profile-to-solid, since
    there is no direct "extrude these points" API. The scratch polyline and
    region are deleted once the solid exists; only the solid remains in the
    drawing.
    """
    if len(profile_polyline) < 3:
        return _error("add_extruded_solid requires at least 3 profile points")
    if is_dry_run_enabled():
        return _dry_run_result(
            "add_extruded_solid", points=[list(p) for p in profile_polyline], height=height
        )
    with _lock:
        poly = None
        region = None
        try:
            ms = _model_space()
            poly = ms.AddLightWeightPolyline(_variant_points(profile_polyline))
            poly.Closed = True
            regions = ms.AddRegion(_variant_object_array([poly]))
            region = regions[0]
            solid = ms.AddExtrudedSolid(region, float(height), 0.0)
            return _ok(
                op="add_extruded_solid",
                handle=str(solid.Handle),
                points=[list(p) for p in profile_polyline],
                height=height,
            )
        except AutoCADConnectionError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(f"add_extruded_solid failed: {exc}")
        finally:
            for scratch in (poly, region):
                try:
                    if scratch is not None:
                        scratch.Delete()
                except Exception:  # noqa: BLE001
                    pass


def run_autolisp_command(lisp_string: str) -> str:
    """Dispatch a raw AutoLISP/command string via ``ActiveDocument.SendCommand``.

    ``SendCommand`` is asynchronous and fire-and-forget from COM's point of
    view — it queues keystrokes into AutoCAD's command line exactly as if
    typed, so it cannot report the command's own success/failure. Callers
    that need a verified result should follow up with
    ``dana.tools.cad_vision.verify_cad_rendering`` rather than trust this
    observation alone.
    """
    text = (lisp_string or "").strip()
    if not text:
        return _error("run_autolisp_command requires a non-empty command string")
    # SendCommand needs a trailing space/Enter to submit the queued line;
    # .strip() above always removes it, so always re-append one.
    text += " "
    if is_dry_run_enabled():
        return _dry_run_result("run_autolisp_command", command=lisp_string)
    with _lock:
        try:
            doc = get_active_document()
            doc.SendCommand(text)
            return _ok(op="run_autolisp_command", command=lisp_string)
        except AutoCADConnectionError as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001
            return _error(f"run_autolisp_command failed: {exc}")


__all__ = (
    "AutoCADConnectionError",
    "add_circle",
    "add_extruded_solid",
    "add_line",
    "add_polyline",
    "get_active_document",
    "get_application",
    "run_autolisp_command",
)
