"""Unit tests for the roadmap-deployed file_jail_enforcer plugin."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from dana.tools.broker import get_broker, reload_broker_registry
from dana.tools.plugins.file_jail_enforcer import (
    DOCS_JAIL,
    file_jail_enforcer,
    handle_tool_call,
    resolve_jailed_path,
)
from dana.tools.schema import ToolCall


class FileJailEnforcerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        DOCS_JAIL.mkdir(parents=True, exist_ok=True)
        cls.sample = DOCS_JAIL / "sample_notes.txt"
        if not cls.sample.is_file():
            cls.sample.write_text(
                "Dana Local Docs Jail\nALPHA-42\n",
                encoding="utf-8",
            )

    def test_read_jailed_text(self) -> None:
        t0 = time.perf_counter()
        result = file_jail_enforcer("sample_notes.txt")
        ms = (time.perf_counter() - t0) * 1000.0
        self.assertTrue(result["ok"], result)
        self.assertIn("ALPHA-42", result["text"])
        self.assertEqual(result["path"], "sample_notes.txt")
        self.assertLess(ms, 100.0, f"latency too high: {ms:.2f} ms")
        type(self).last_latency_ms = ms  # type: ignore[attr-defined]

    def test_blocks_traversal(self) -> None:
        result = file_jail_enforcer("../agent.py")
        self.assertFalse(result["ok"])
        self.assertIn("traversal", (result.get("error") or "").lower())

    def test_blocks_absolute(self) -> None:
        result = file_jail_enforcer(r"C:\Windows\System32\drivers\etc\hosts")
        self.assertFalse(result["ok"])

    def test_resolve_stays_in_jail(self) -> None:
        path = resolve_jailed_path("sample_notes.txt")
        self.assertTrue(str(path).startswith(str(DOCS_JAIL)))

    def test_broker_dispatch(self) -> None:
        reload_broker_registry()
        broker = get_broker()
        self.assertIn("file_jail_enforcer", broker.registry)
        call = ToolCall(
            tool_id="file_jail_enforcer",
            arguments={"path": "sample_notes.txt"},
            source_lang="en",
        )
        obs = broker.dispatch(call, handlers={})
        self.assertTrue(str(obs).startswith("OK: file_jail_enforcer"))
        self.assertIn("ALPHA-42", str(obs))

    def test_missing_path(self) -> None:
        obs = handle_tool_call(ToolCall(tool_id="file_jail_enforcer", arguments={}))
        self.assertTrue(str(obs).startswith("ERROR: missing path"))

    def test_missing_file(self) -> None:
        result = file_jail_enforcer("does_not_exist_xyz.txt")
        self.assertFalse(result["ok"])
        self.assertIn("not found", (result.get("error") or "").lower())


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FileJailEnforcerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    latency = getattr(FileJailEnforcerTests, "last_latency_ms", None)
    if latency is not None:
        print(f"\n[METRICS] file_jail_enforcer core latency: {latency:.3f} ms")
    raise SystemExit(0 if result.wasSuccessful() else 1)
