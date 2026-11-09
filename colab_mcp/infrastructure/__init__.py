"""Infrastructure Layer Router.

Exposes external adapters for process execution, configuration persistence,
and browser communication.
"""

from colab_mcp.infrastructure.browser_adapter import BrowserAdapter
from colab_mcp.infrastructure.config_repository import ConfigRepository
from colab_mcp.infrastructure.process_adapter import ProcessAdapter

__all__ = [
    "BrowserAdapter",
    "ConfigRepository",
    "ProcessAdapter",
]
