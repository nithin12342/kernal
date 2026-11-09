"""Domain Layer Router.

Exposes domain models, entities, and constants to upper layers while
enforcing encapsulation of internal domain structures.
"""

from colab_mcp.domain.constants import (
    COLAB_MCP_REPO,
    COLAB_MCP_SERVER_KEY,
    COLAB_URL,
    DEFAULT_TIMEOUT_MS,
    MCP_CONFIG_PATH,
    SCHEMA_CACHE_DIR,
    TARGET_PROCESS_NAMES,
)
from colab_mcp.domain.models import (
    ConfigurationStatus,
    PrerequisiteStatus,
    SystemStatusReport,
)

__all__ = [
    "COLAB_MCP_REPO",
    "COLAB_MCP_SERVER_KEY",
    "COLAB_URL",
    "DEFAULT_TIMEOUT_MS",
    "MCP_CONFIG_PATH",
    "SCHEMA_CACHE_DIR",
    "TARGET_PROCESS_NAMES",
    "ConfigurationStatus",
    "PrerequisiteStatus",
    "SystemStatusReport",
]
