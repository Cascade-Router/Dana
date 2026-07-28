from dana.agentic import _obs_fallback


def test_forge_ok_not_fallback() -> None:
    spoken = _obs_fallback(
        "OK: Tool Forge forged and hot-loaded `abc`.",
        reply_lang="en",
    )
    assert "couldn't finish" not in spoken.lower()
    assert "abc" in spoken or "Done" in spoken
    batch = _obs_fallback(
        "OK: Tool Forge batch status=loaded loaded=['a', 'b', 'c'].",
        reply_lang="en",
    )
    assert "couldn't finish" not in batch.lower()
    assert "3" in batch or "a" in batch
    print("[PASS] forge OK spoken")


if __name__ == "__main__":
    test_forge_ok_not_fallback()
    print("OK")
