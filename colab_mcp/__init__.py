"""Google Colab MCP Manager (Domain-Driven Design Architecture).

This top-level package acts as a pure B+ tree routing node, exposing
the public API boundary to external consumers.
"""

from colab_mcp.application.service import ColabApplicationService
from colab_mcp.domain.models import (
    ConfigurationStatus,
    PrerequisiteStatus,
    SystemStatusReport,
)

__all__ = [
    "ColabApplicationService",
    "ConfigurationStatus",
    "PrerequisiteStatus",
    "SystemStatusReport",
    "setup",
    "reset",
    "status",
    "launch",
]


def setup() -> bool:
    """Public API: Configure Colab MCP and prerequisites."""
    return ColabApplicationService.setup_environment()


def reset():
    """Public API: Reset active processes and session caches."""
    return ColabApplicationService.reset_active_session()


def status() -> SystemStatusReport:
    """Public API: Retrieve complete system status report."""
    return ColabApplicationService.inspect_system_status()


def launch() -> bool:
    """Public API: Open Google Colab in browser."""
    return ColabApplicationService.launch_browser()
