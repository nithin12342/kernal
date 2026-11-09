"""Domain Constants Module.

Contains immutable domain values, paths, configuration targets, and URLs.
This module is a pure domain leaf node with zero external dependencies.
"""

from pathlib import Path

# Filesystem and System Configuration Paths
HOME_DIR = Path.home()
GEMINI_CONFIG_DIR = HOME_DIR / ".gemini" / "config"
MCP_CONFIG_PATH = GEMINI_CONFIG_DIR / "mcp_config.json"
SCHEMA_CACHE_DIR = HOME_DIR / ".gemini" / "antigravity" / "mcp" / "colab-mcp"

# Google Colab MCP Server Metadata
COLAB_MCP_REPO = "git+https://github.com/googlecolab/colab-mcp"
COLAB_MCP_SERVER_KEY = "colab-mcp"
DEFAULT_TIMEOUT_MS = 30000

# Browser URLs
COLAB_URL = "https://colab.research.google.com"

# Managed Operating System Process Names
TARGET_PROCESS_NAMES = ["colab-mcp", "uv", "uvx"]
