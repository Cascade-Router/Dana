"""Tests for dana.plugins.vision.ocr_grounding. The real easyocr.Reader is
always mocked — instantiating one loads/downloads real models, which these
tests must never do (confirmed slow and network-dependent live).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dana.plugins.os import file_system
from dana.plugins.vision import ocr_grounding


@pytest.fixture(autouse=True)
def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "agent_workspace"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(file_system, "_SANDBOX_ROOT", root)
    return root


@pytest.fixture(autouse=True)
def _reset_reader_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reader is a module-level singleton (see ocr_grounding._get_reader's
    own docstring for why) -- reset it per test so one test's mocked/absent
    reader never leaks into the next.
    """
    monkeypatch.setattr(ocr_grounding, "_reader", None)


class _FakeReader:
    def __init__(self, texts: list[str] | Exception) -> None:
        self._texts = texts

    def readtext(self, path: str, detail: int = 0) -> list[str]:
        if isinstance(self._texts, Exception):
            raise self._texts
        return self._texts


def _mock_reader(monkeypatch: pytest.MonkeyPatch, texts: list[str] | Exception) -> None:
    monkeypatch.setattr(ocr_grounding, "_get_reader", lambda: _FakeReader(texts))


def _write_png(sandbox: Path, name: str) -> None:
    (sandbox / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


# --------------------------------------------------------------------------
# _infer_view_label
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("bracket_front.png", "FRONT"),
        ("bracket_top.png", "TOP"),
        ("bracket_right.png", "RIGHT"),
        ("shaft_side.png", "SIDE"),
        ("front-view.jpg", "FRONT"),
        ("Front View.jpg", "FRONT"),
    ],
)
def test_infer_view_label_recognizes_known_keywords(filename: str, expected: str) -> None:
    assert ocr_grounding._infer_view_label(Path(filename), index=0) == expected


def test_infer_view_label_falls_back_to_position_for_arbitrary_filenames() -> None:
    """A real chat-upload file has no reason to follow this project's own
    synthetic naming convention -- guessing a view name from an arbitrary
    filename would be confidently WRONG rather than just unlabeled, so this
    must fall back rather than pattern-match something coincidental.
    """
    assert ocr_grounding._infer_view_label(Path("IMG_1234.jpg"), index=2) == "VIEW_3"
    assert ocr_grounding._infer_view_label(Path("photo.png"), index=0) == "VIEW_1"


# --------------------------------------------------------------------------
# _repair_and_filter
# --------------------------------------------------------------------------


def test_repair_and_filter_fixes_space_for_decimal_misread() -> None:
    # Confirmed live: EasyOCR misread "60.0" (this project's own stamped
    # label) as "60 0" -- the decimal point rendered as/merged into a space.
    assert ocr_grounding._repair_and_filter("60 0") == "60.0"


def test_repair_and_filter_rejects_bare_digit_fragments() -> None:
    # Confirmed live: EasyOCR produced bare "8" and "3" fragments (likely
    # misreads of rotated labels) alongside the real "60 0" -- a looser
    # filter (any digit at all) let these through and the VLM later stuffed
    # them verbatim into an unrelated fabricated primitive.
    assert ocr_grounding._repair_and_filter("8") is None
    assert ocr_grounding._repair_and_filter("3") is None


def test_repair_and_filter_accepts_radius_and_diameter_prefixes() -> None:
    assert ocr_grounding._repair_and_filter("R5") == "R5"
    assert ocr_grounding._repair_and_filter("Ø12.5") == "Ø12.5"


def test_repair_and_filter_rejects_unrelated_text() -> None:
    assert ocr_grounding._repair_and_filter("hello") is None


# --------------------------------------------------------------------------
# extract_blueprint_dimensions
# --------------------------------------------------------------------------


def test_extract_blueprint_dimensions_groups_by_inferred_view(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "bracket_front.png")
    _write_png(_sandbox, "bracket_top.png")

    def fake_get_reader():
        class _Reader:
            def readtext(self, path: str, detail: int = 0) -> list[str]:
                return ["60 0", "8"] if "front" in path else ["60 0", "40.0"]

        return _Reader()

    monkeypatch.setattr(ocr_grounding, "_get_reader", fake_get_reader)

    result = ocr_grounding.extract_blueprint_dimensions(["bracket_front.png", "bracket_top.png"])

    assert result == {"FRONT": ["60.0"], "TOP": ["40.0", "60.0"]}


def test_extract_blueprint_dimensions_returns_empty_dict_when_reader_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ocr_grounding, "_get_reader", lambda: None)
    assert ocr_grounding.extract_blueprint_dimensions(["anything.png"]) == {}


def test_extract_blueprint_dimensions_missing_file_gets_empty_list_not_a_crash(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_reader(monkeypatch, texts=["60.0"])
    result = ocr_grounding.extract_blueprint_dimensions(["missing_front.png"])
    assert result == {"FRONT": []}


def test_extract_blueprint_dimensions_ocr_exception_gets_empty_list_not_a_crash(
    _sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_png(_sandbox, "bracket_front.png")
    _mock_reader(monkeypatch, texts=RuntimeError("OCR engine crashed"))
    result = ocr_grounding.extract_blueprint_dimensions(["bracket_front.png"])
    assert result == {"FRONT": []}


def test_get_reader_returns_none_when_easyocr_not_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "easyocr":
            raise ImportError("no module named easyocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ocr_grounding._get_reader() is None
