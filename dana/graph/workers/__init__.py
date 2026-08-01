"""Specialized worker nodes for supervisor sub-agent routing."""

from dana.graph.workers.os_worker import (
    OS_WORKER_NODE,
    OS_WORKER_SYSTEM_PROMPT,
    make_os_worker_node,
    os_worker_node,
    route_after_executor,
    should_route_to_os_worker,
)

__all__ = (
    "OS_WORKER_NODE",
    "OS_WORKER_SYSTEM_PROMPT",
    "make_os_worker_node",
    "os_worker_node",
    "route_after_executor",
    "should_route_to_os_worker",
)
