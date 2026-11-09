"""Colab MCP Control CLI Runner.

Top-level command runner delegating to the DDD interfaces layer.

Usage:
    python colab_ctl.py status   # Inspect environment and active processes
    python colab_ctl.py setup    # Install uv, cache package, register in mcp_config.json
    python colab_ctl.py reset    # Stop active sessions to switch Google accounts cleanly
    python colab_ctl.py launch   # Open Colab in browser to select account / GPU runtime
"""

import sys
from colab_mcp.interfaces.cli import main

if __name__ == "__main__":
    main()
