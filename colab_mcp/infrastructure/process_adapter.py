"""Infrastructure Process Adapter Module.

Handles OS-level process management, command execution, and version resolution.
"""

import logging
import os
import shutil
import subprocess
import sys
from typing import List, Optional

logger = logging.getLogger("colab_mcp.infrastructure.process")


class ProcessAdapter:
    """Interacts with host operating system processes and executable environments."""

    @staticmethod
    def get_executable_version(name: str) -> Optional[str]:
        """Query executable version via CLI '--version' command."""
        exe_path = shutil.which(name)
        if not exe_path:
            return None
        try:
            res = subprocess.run(
                [name, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception as exc:
            logger.debug(f"Failed to get version for {name}: {exc}")
        return f"Detected at {exe_path}"

    @staticmethod
    def install_pip_package(package_name: str) -> bool:
        """Install a package into the active Python environment using pip."""
        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception as exc:
            logger.error(f"Failed to pip install {package_name}: {exc}")
            return False

    @staticmethod
    def execute_prebuild(uvx_executable: str, repo_spec: str) -> bool:
        """Run uvx build to compile and cache the remote repository."""
        try:
            res = subprocess.run(
                [uvx_executable, repo_spec, "--help"],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception as exc:
            logger.error(f"Prebuild execution failed: {exc}")
            return False

    @staticmethod
    def terminate_processes(names: List[str]) -> List[str]:
        """Forcefully terminate named processes across Windows or POSIX."""
        terminated: List[str] = []
        if os.name == "nt":
            for name in names:
                try:
                    res = subprocess.run(
                        ["taskkill", "/F", "/IM", f"{name}.exe"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    if res.returncode == 0:
                        terminated.append(name)
                except Exception as exc:
                    logger.debug(f"Taskkill error for {name}: {exc}")
        return terminated

    @staticmethod
    def list_running_processes(names: List[str]) -> List[str]:
        """Detect which of the specified processes are actively running."""
        running: List[str] = []
        if os.name == "nt":
            for name in names:
                try:
                    res = subprocess.run(
                        ["tasklist", "/FI", f"IMAGENAME eq {name}.exe"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if f"{name}.exe" in res.stdout:
                        running.append(name)
                except Exception as exc:
                    logger.debug(f"Tasklist query failed for {name}: {exc}")
        return running
