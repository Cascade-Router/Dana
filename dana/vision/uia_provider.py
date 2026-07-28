"""Win32 UI Automation provider for native control bounding boxes.

Searches the active window control tree by name, automation_id, or
control_type. Returns Florence-compatible ``[0, 1000]^4`` normalized bounds.

UIA backends are optional: ``pywinauto``, ``uiautomation``, or ``ctypes`` /
``comtypes`` when present. CI without Windows UIA can still import this module;
live queries then return ``None``. Tests inject a fake control tree.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Iterable, Optional, Sequence

_log = logging.getLogger("dana.vision.uia")

# Control types that rarely expose stable text labels (canvas / drawing surfaces).
_CANVAS_LIKE = frozenset(
    {
        "image",
        "pane",
        "custom",
        "document",
        "canvas",
        "thumb",
        "separator",
    }
)

NormBBox = list[float]  # [x1, y1, x2, y2] in [0, 1000]


def _clamp01k(v: float) -> float:
    return max(0.0, min(1000.0, float(v)))


def screen_rect_to_norm1000(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    screen_wh: tuple[int, int],
) -> NormBBox | None:
    """Map absolute screen pixels → Florence ``[0, 1000]`` xyxy."""
    sw, sh = int(screen_wh[0]), int(screen_wh[1])
    if sw <= 0 or sh <= 0:
        return None
    x1 = _clamp01k(left / sw * 1000.0)
    y1 = _clamp01k(top / sh * 1000.0)
    x2 = _clamp01k(right / sw * 1000.0)
    y2 = _clamp01k(bottom / sh * 1000.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _norm_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_canvas_or_unannotated(node: dict[str, Any]) -> bool:
    """True when the element has no usable label or is a canvas-like surface."""
    name = _norm_label(node.get("name") or node.get("Name"))
    auto_id = _norm_label(node.get("automation_id") or node.get("AutomationId"))
    ctype = _norm_label(node.get("control_type") or node.get("ControlType"))
    if node.get("unannotated") or node.get("is_canvas"):
        return True
    if ctype in _CANVAS_LIKE and not name and not auto_id:
        return True
    if not name and not auto_id and not ctype:
        return True
    return False


def _node_matches(node: dict[str, Any], target: str) -> bool:
    q = _norm_label(target)
    if not q:
        return False
    fields = (
        node.get("name"),
        node.get("Name"),
        node.get("automation_id"),
        node.get("AutomationId"),
        node.get("control_type"),
        node.get("ControlType"),
    )
    for field in fields:
        text = _norm_label(field)
        if not text:
            continue
        if q == text or q in text or text in q:
            return True
    return False


def _bounds_from_node(
    node: dict[str, Any],
    *,
    screen_wh: tuple[int, int] | None,
) -> NormBBox | None:
    """Accept pre-normalized ``bounds_norm`` or pixel ``bounds`` / ``rect``."""
    norm = node.get("bounds_norm") or node.get("xyxy_norm")
    if norm is not None and len(list(norm)) >= 4:
        vals = [float(v) for v in list(norm)[:4]]
        x1, y1, x2, y2 = vals
        if x2 > x1 and y2 > y1:
            return [_clamp01k(x1), _clamp01k(y1), _clamp01k(x2), _clamp01k(y2)]
        return None

    rect = node.get("bounds") or node.get("rect") or node.get("rectangle")
    if rect is None:
        return None
    try:
        if isinstance(rect, dict):
            left = float(rect.get("left", rect.get("x", 0)))
            top = float(rect.get("top", rect.get("y", 0)))
            right = float(
                rect.get(
                    "right",
                    left + float(rect.get("width", 0)),
                )
            )
            bottom = float(
                rect.get(
                    "bottom",
                    top + float(rect.get("height", 0)),
                )
            )
        else:
            vals = [float(v) for v in list(rect)[:4]]
            left, top, right, bottom = vals
            # width/height form
            if right < left or bottom < top:
                right = left + abs(right)
                bottom = top + abs(bottom)
    except Exception:  # noqa: BLE001
        return None

    wh = screen_wh
    if wh is None:
        # Treat as already-normalized if span looks like Florence space.
        span = max(abs(left), abs(top), abs(right), abs(bottom))
        if span <= 1000.5:
            return [
                _clamp01k(left),
                _clamp01k(top),
                _clamp01k(right),
                _clamp01k(bottom),
            ]
        return None
    return screen_rect_to_norm1000(left, top, right, bottom, screen_wh=wh)


def _flatten_tree(nodes: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        out.append(node)
        children = node.get("children") or node.get("Children") or []
        if children:
            out.extend(_flatten_tree(children))
    return out


def _default_screen_wh() -> tuple[int, int] | None:
    try:
        from dana.tracker import primary_monitor_geometry

        mon = primary_monitor_geometry()
        if mon and mon.get("width") and mon.get("height"):
            return int(mon["width"]), int(mon["height"])
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:  # noqa: BLE001
            return None
    return None


def _live_control_tree_via_pywinauto() -> list[dict[str, Any]] | None:
    try:
        from pywinauto import Desktop  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        win = Desktop(backend="uia").get_active()
        if win is None:
            return None
        nodes: list[dict[str, Any]] = []

        def _walk(ctrl: Any, depth: int = 0) -> None:
            if depth > 24:
                return
            try:
                info = ctrl.element_info
                rect = info.rectangle
                nodes.append(
                    {
                        "name": getattr(info, "name", "") or "",
                        "automation_id": getattr(info, "automation_id", "") or "",
                        "control_type": str(getattr(info, "control_type", "") or ""),
                        "bounds": (
                            float(rect.left),
                            float(rect.top),
                            float(rect.right),
                            float(rect.bottom),
                        ),
                    }
                )
            except Exception:  # noqa: BLE001
                return
            try:
                for child in ctrl.children():
                    _walk(child, depth + 1)
            except Exception:  # noqa: BLE001
                return

        _walk(win)
        return nodes
    except Exception as exc:  # noqa: BLE001
        _log.debug("pywinauto UIA walk failed: %s", exc)
        return None


def _live_control_tree_via_uiautomation() -> list[dict[str, Any]] | None:
    try:
        import uiautomation as auto  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        root = auto.GetForegroundControl()
        if root is None:
            return None
        nodes: list[dict[str, Any]] = []

        def _walk(ctrl: Any, depth: int = 0) -> None:
            if ctrl is None or depth > 24:
                return
            try:
                rect = ctrl.BoundingRectangle
                nodes.append(
                    {
                        "name": str(ctrl.Name or ""),
                        "automation_id": str(ctrl.AutomationId or ""),
                        "control_type": str(
                            getattr(ctrl, "ControlTypeName", "") or ""
                        ),
                        "bounds": (
                            float(rect.left),
                            float(rect.top),
                            float(rect.right),
                            float(rect.bottom),
                        ),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                child = ctrl.GetFirstChildControl()
                while child is not None:
                    _walk(child, depth + 1)
                    child = child.GetNextSiblingControl()
            except Exception:  # noqa: BLE001
                return

        _walk(root)
        return nodes
    except Exception as exc:  # noqa: BLE001
        _log.debug("uiautomation walk failed: %s", exc)
        return None


def _live_control_tree_via_ctypes() -> list[dict[str, Any]] | None:
    """Minimal foreground-window walk via UIAutomation COM (Windows only)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    try:
        import comtypes  # noqa: F401
        import comtypes.client
    except ImportError:
        # Without comtypes we can still report the foreground window rect only.
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None
            length = int(user32.GetWindowTextLengthW(hwnd))
            buf = ctypes.create_unicode_buffer(length + 2)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            rect = wintypes.RECT()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None
            return [
                {
                    "name": (buf.value or "").strip(),
                    "automation_id": "",
                    "control_type": "Window",
                    "bounds": (
                        float(rect.left),
                        float(rect.top),
                        float(rect.right),
                        float(rect.bottom),
                    ),
                }
            ]
        except Exception as exc:  # noqa: BLE001
            _log.debug("ctypes window rect failed: %s", exc)
            return None

    # Soft COM path — best-effort; failures stay None for Florence fallback.
    try:
        uia = comtypes.client.CreateObject(
            "{ff48dba4-60ef-4201-aa87-54103eef594e}",
            interface=None,
        )
        # Without generated IUIAutomation typelib bindings, skip deep walk.
        _log.debug("comtypes UIA present but typelib walk not wired; skip")
        _ = uia
        return None
    except Exception as exc:  # noqa: BLE001
        _log.debug("comtypes UIA create failed: %s", exc)
        return None


def fetch_live_control_tree() -> list[dict[str, Any]]:
    """Best-effort live UIA tree; empty list when backends unavailable."""
    for fetcher in (
        _live_control_tree_via_pywinauto,
        _live_control_tree_via_uiautomation,
        _live_control_tree_via_ctypes,
    ):
        try:
            tree = fetcher()
        except Exception as exc:  # noqa: BLE001
            _log.debug("%s failed: %s", fetcher.__name__, exc)
            continue
        if tree:
            return list(tree)
    return []


class Win32UIAProvider:
    """Query native Win32 UI Automation bounds for a target label.

    Parameters
    ----------
    control_tree:
        Injectable list of control dicts (tests / offline). Each node may
        include ``name``, ``automation_id``, ``control_type``, and either
        ``bounds_norm`` (``[0,1000]``) or pixel ``bounds`` / ``rect``.
    window:
        Optional injectable window handle / descriptor (reserved for backends).
    screen_wh:
        Screen width/height used to normalize pixel bounds. Defaults to primary
        monitor geometry when available.
    tree_fetcher:
        Injectable live-tree callback (defaults to ``fetch_live_control_tree``).
    """

    def __init__(
        self,
        *,
        control_tree: Sequence[dict[str, Any]] | None = None,
        window: Any = None,
        screen_wh: tuple[int, int] | None = None,
        tree_fetcher: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.control_tree: list[dict[str, Any]] | None = (
            list(control_tree) if control_tree is not None else None
        )
        self.window = window
        self.screen_wh = screen_wh
        self.tree_fetcher = tree_fetcher or fetch_live_control_tree

    def _resolve_tree(self) -> list[dict[str, Any]]:
        if self.control_tree is not None:
            return _flatten_tree(self.control_tree)
        try:
            return _flatten_tree(self.tree_fetcher() or [])
        except Exception as exc:  # noqa: BLE001
            _log.debug("UIA tree_fetcher failed: %s", exc)
            return []

    def find_element_bounds(self, target_label: str) -> Optional[NormBBox]:
        """Return ``[x1,y1,x2,y2]`` in ``[0,1000]`` or ``None`` if not found.

        Returns ``None`` for missing matches, unannotated nodes, and canvas-like
        surfaces without a usable name / automation id.
        """
        target = str(target_label or "").strip()
        if not target:
            return None

        nodes = self._resolve_tree()
        if not nodes:
            return None

        wh = self.screen_wh or _default_screen_wh()
        for node in nodes:
            if not _node_matches(node, target):
                continue
            if _is_canvas_or_unannotated(node):
                return None
            box = _bounds_from_node(node, screen_wh=wh)
            if box is not None:
                return box
        return None
