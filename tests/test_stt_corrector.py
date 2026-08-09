"""STT vocabulary middleware tests (Notepad phonetic repairs + Titan codename)."""

from __future__ import annotations

from dana.tools.stt_corrector import (
    correct_stt,
    correct_titan_codename_stt,
    reload_vocabulary,
    strip_trailing_punctuation_hallucinations,
)


def test_notepad_phonetic_corrections() -> None:
    reload_vocabulary()
    assert correct_stt("open no it's bad") == "open Notepad"
    assert correct_stt("notes pad") == "Notepad"
    assert correct_stt("notes, bad") == "Notepad"
    assert correct_stt("Open Notes Pad please").lower() == "open notepad please"
    assert "Notepad" in correct_stt("Open Notes Pad please")

    # Unrelated text unchanged.
    assert correct_stt("what time is it") == "what time is it"
    print("[PASS] Notepad STT vocabulary corrections")


def test_titan_codename_corrections() -> None:
    reload_vocabulary()
    assert correct_stt("activate the json initiative") == "activate the Titan initiative"
    assert correct_stt("Dana, activate the Jason initiative") == (
        "Dana, activate the Titan initiative"
    )
    assert correct_stt("start the json protocol") == "start the Titan Protocol"
    assert correct_stt("run the json supervisor") == "run the Titan supervisor"
    # Unrelated JSON/file context left alone.
    assert correct_stt("parse this config") == "parse this config"
    print("[PASS] Titan codename STT corrections")


def test_titan_context_guard() -> None:
    assert correct_titan_codename_stt("activate the JSON") == "activate the Titan"
    assert correct_titan_codename_stt("hello world") == "hello world"
    print("[PASS] Titan context guard")


def test_trailing_punctuation_hallucinations_are_stripped() -> None:
    assert strip_trailing_punctuation_hallucinations("relationship reflection!!!") == (
        "relationship reflection"
    )
    assert correct_stt("self-indulcated data code...") == "self-indulcated data code"
    assert correct_stt("what time is it?") == "what time is it?"
    print("[PASS] trailing punctuation hallucination cleanup")


if __name__ == "__main__":
    test_notepad_phonetic_corrections()
    test_titan_codename_corrections()
    test_titan_context_guard()
    test_trailing_punctuation_hallucinations_are_stripped()
    print("OK")
