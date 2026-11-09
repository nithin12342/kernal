"""Application Use Cases Module.

Implements discrete, testable business use cases for Colab MCP operations.
"""

import shutil
import sys
from typing import List, Tuple

from colab_mcp.domain import (
    COLAB_MCP_REPO,
    COLAB_MCP_SERVER_KEY,
    COLAB_URL,
    DEFAULT_TIMEOUT_MS,
    MCP_CONFIG_PATH,
    SCHEMA_CACHE_DIR,
    TARGET_PROCESS_NAMES,
    ConfigurationStatus,
    PrerequisiteStatus,
    SystemStatusReport,
)
from colab_mcp.infrastructure import (
    BrowserAdapter,
    ConfigRepository,
    ProcessAdapter,
)


class SetupUseCase:
    """Orchestrates complete setup: uv check, config writing, and pre-caching."""

    @staticmethod
    def execute() -> bool:
        """Execute the setup workflow."""
        if not (shutil.which("uv") and shutil.which("uvx")):
            if not ProcessAdapter.install_pip_package("uv"):
                return False

        # Register server in mcp_config.json
        config_data = ConfigRepository.load_json(MCP_CONFIG_PATH)
        servers = config_data.get("mcpServers", {})
        servers[COLAB_MCP_SERVER_KEY] = {
            "command": "uvx",
            "args": [COLAB_MCP_REPO],
            "timeout": DEFAULT_TIMEOUT_MS,
        }
        config_data["mcpServers"] = servers
        if not ConfigRepository.save_json(MCP_CONFIG_PATH, config_data):
            return False

        # Pre-build cache
        uvx_cmd = shutil.which("uvx") or "uvx"
        ProcessAdapter.execute_prebuild(uvx_cmd, COLAB_MCP_REPO)
        return True


class ResetSessionUseCase:
    """Terminates active processes and clears caches to facilitate account switching."""

    @staticmethod
    def execute() -> Tuple[List[str], bool]:
        """Execute the reset workflow."""
        terminated = ProcessAdapter.terminate_processes(TARGET_PROCESS_NAMES)
        cache_cleared = ConfigRepository.clear_directory(SCHEMA_CACHE_DIR)
        return terminated, cache_cleared


class InspectStatusUseCase:
    """Collects and synthesizes environmental, configuration, and process status."""

    @staticmethod
    def execute() -> SystemStatusReport:
        """Inspect and return complete system health."""
        prereqs = PrerequisiteStatus(
            python_installed=True,
            python_version=sys.version.split()[0],
            git_installed=ProcessAdapter.get_executable_version("git") is not None,
            git_version=ProcessAdapter.get_executable_version("git"),
            uv_installed=ProcessAdapter.get_executable_version("uv") is not None,
            uv_version=ProcessAdapter.get_executable_version("uv"),
        )

        config_data = ConfigRepository.load_json(MCP_CONFIG_PATH)
        servers = config_data.get("mcpServers", {})
        entry = servers.get(COLAB_MCP_SERVER_KEY)

        conf_status = ConfigurationStatus(
            config_exists=MCP_CONFIG_PATH.is_file(),
            is_registered=entry is not None,
            server_command=entry.get("command") if entry else None,
            server_args=entry.get("args") if entry else None,
        )

        running = ProcessAdapter.list_running_processes(TARGET_PROCESS_NAMES)
        ready = (
            prereqs.git_installed
            and prereqs.uv_installed
            and conf_status.is_registered
        )

        return SystemStatusReport(
            prerequisites=prereqs,
            config=conf_status,
            running_processes=running,
            is_ready=ready,
        )


class LaunchColabUseCase:
    """Launches the user's browser directed to Google Colab."""

    @staticmethod
    def execute() -> bool:
        """Open Google Colab URL."""
        return BrowserAdapter.open_url(COLAB_URL)
