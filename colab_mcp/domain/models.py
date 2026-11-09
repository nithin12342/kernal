"""Domain Models Module.

Defines the core business entities, value objects, and status data models
for Google Colab MCP lifecycle management.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PrerequisiteStatus:
    """Value object representing host system software prerequisites."""

    python_installed: bool
    python_version: str
    git_installed: bool
    git_version: Optional[str]
    uv_installed: bool
    uv_version: Optional[str]


@dataclass(frozen=True)
class ConfigurationStatus:
    """Value object representing MCP JSON configuration state."""

    config_exists: bool
    is_registered: bool
    server_command: Optional[str]
    server_args: Optional[List[str]]


@dataclass(frozen=True)
class SystemStatusReport:
    """Composite domain aggregate reporting overall system health and readiness."""

    prerequisites: PrerequisiteStatus
    config: ConfigurationStatus
    running_processes: List[str]
    is_ready: bool
