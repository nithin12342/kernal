"""Rust Compiler Clippy Linter Runner.

Runs cargo clippy with strict denial of all warnings to guarantee code hygiene.
"""

import subprocess
from pathlib import Path
from typing import Tuple


class RustClippyRunner:
    """Executes cargo clippy on the Rust kernel crate."""

    @staticmethod
    def run_clippy(crate_dir: Path) -> Tuple[bool, str]:
        """Runs `cargo clippy --all-targets -- -D warnings`."""
        if not crate_dir.exists():
            return False, f"Crate directory {crate_dir} does not exist"

        try:
            res = subprocess.run(
                ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
                cwd=crate_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            success = (res.returncode == 0)
            output = res.stdout + "\n" + res.stderr
            return success, output.strip()
        except Exception as exc:
            return False, f"Clippy execution failed: {exc}"
