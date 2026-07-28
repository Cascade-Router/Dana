"""Security reviewer must not false-reject AST-clean psutil tools."""

from __future__ import annotations

from dana.swarm.tool_forge_graph import security_reviewer_agent
from dana.swarm.tool_forge_template import assemble_forged_tool


def test_security_short_circuit_approves_psutil() -> None:
    code = assemble_forged_tool(
        tool_name="check_cpu_ram",
        docstring="Report CPU and RAM percent.",
        python_code=(
            "cpu = psutil.cpu_percent(interval=0.1)\n"
            "ram = psutil.virtual_memory().percent\n"
            "alert = cpu > 85 or ram > 85\n"
            "return f'cpu={cpu} ram={ram} alert={alert}'"
        ),
        description="CPU/RAM check",
    )
    out = security_reviewer_agent(
        {
            "code": code,
            "query": "build a tool that checks CPU and RAM",
            "tool_name": "check_cpu_ram",
            "revisions": 0,
            "history": [],
        }
    )
    assert out["status"] == "APPROVED", out
    print("[PASS] security short-circuit APPROVED for psutil tool")


if __name__ == "__main__":
    test_security_short_circuit_approves_psutil()
    print("OK")
