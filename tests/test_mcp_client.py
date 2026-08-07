"""MCP client helpers (discovery formatting; no live servers required)."""

from __future__ import annotations

from dana.mcp.client import MCPTool, format_mcp_tools_block


def test_format_mcp_tools_block_empty() -> None:
    block = format_mcp_tools_block([])
    assert "MCP Tools" in block
    assert "none discovered" in block.lower() or "DONNA_MCP_SERVERS" in block


def test_format_mcp_tools_block_lists_tools() -> None:
    tools = [
        MCPTool(
            name="ros2.list_nodes",
            description="List active ROS2 nodes",
            input_schema={"properties": {"namespace": {"type": "string"}}},
            server="ros",
        )
    ]
    block = format_mcp_tools_block(tools)
    assert "ros2.list_nodes" in block
    assert "namespace" in block
