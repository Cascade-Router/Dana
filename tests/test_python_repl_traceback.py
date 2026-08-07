from dana.tools.system_repl import python_repl


def test_python_repl_reports_full_traceback_for_zero_division_error() -> None:
    out = python_repl("1/0")
    assert "--- EXECUTION ERROR ---" in out
    assert "ZeroDivisionError" in out
    assert "File:" in out
    assert "Line:" in out
    assert "Traceback:" in out


def test_python_repl_reports_full_traceback_for_syntax_error() -> None:
    out = python_repl("if True:\n    print('oops'\n")
    assert "--- EXECUTION ERROR ---" in out
    assert "SyntaxError" in out
    assert "File:" in out
    assert "Line:" in out
    assert "Traceback:" in out
