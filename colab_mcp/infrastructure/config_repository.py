"""Infrastructure Configuration Repository Module.

Manages persistent filesystem storage of MCP server declarations and schema cache state.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("colab_mcp.infrastructure.config")


class ConfigRepository:
    """Repository handling read/write operations for MCP JSON configuration files."""

    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        """Safely deserialize a JSON configuration file."""
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except Exception as exc:
            logger.warning(f"Failed to read JSON at {path}: {exc}")
            return {}

    @staticmethod
    def save_json(path: Path, payload: Dict[str, Any]) -> bool:
        """Safely serialize and write payload to a JSON file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            return True
        except Exception as exc:
            logger.error(f"Failed to save JSON to {path}: {exc}")
            return False

    @staticmethod
    def clear_directory(path: Path) -> bool:
        """Safely delete and clear a target directory tree."""
        if not path.exists():
            return False
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception as exc:
            logger.warning(f"Directory deletion failed for {path}: {exc}")
            return False
