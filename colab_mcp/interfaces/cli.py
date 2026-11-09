"""Interface Layer: Command Line Interface.

Provides CLI command parsing and formatted terminal presentation.
"""

import argparse
import sys
from colab_mcp.application import ColabApplicationService


def run_setup(_args: argparse.Namespace) -> None:
    """Handle setup CLI command."""
    print("==> Checking prerequisites and configuring Colab MCP...")
    success = ColabApplicationService.setup_environment()
    if success:
        print("[OK] Colab MCP registered in mcp_config.json and pre-cached!")
    else:
        print("[X] Setup failed. Verify network access and permissions.")


def run_status(_args: argparse.Namespace) -> None:
    """Handle status CLI command."""
    report = ColabApplicationService.inspect_system_status()
    print("\n================ Colab MCP Status ================")
    print(f"Python:       {'[OK]' if report.prerequisites.python_installed else '[X]'} ({report.prerequisites.python_version})")
    print(f"Git:          {'[OK]' if report.prerequisites.git_installed else '[X]'} ({report.prerequisites.git_version or 'Not found'})")
    print(f"uv / uvx:     {'[OK]' if report.prerequisites.uv_installed else '[X]'} ({report.prerequisites.uv_version or 'Not found'})")
    print("--------------------------------------------------")
    print(f"mcp_config:   {'[OK] Found' if report.config.config_exists else '[X] Missing'}")
    print(f"Registered:   {'[OK] Yes' if report.config.is_registered else '[X] No'}")
    if report.config.server_command:
        args_str = " ".join(report.config.server_args or [])
        print(f"Command:      {report.config.server_command} {args_str}")
    print("--------------------------------------------------")
    procs = ", ".join(report.running_processes) if report.running_processes else "None currently active"
    print(f"Processes:    {procs}")
    print(f"Overall:      {'[READY]' if report.is_ready else '[ACTION REQUIRED]'}")
    print("==================================================\n")


def run_reset(_args: argparse.Namespace) -> None:
    """Handle reset CLI command."""
    print("==> Terminating active Colab MCP processes & clearing caches...")
    terminated, cache_cleared = ColabApplicationService.reset_active_session()
    if terminated:
        print(f"[OK] Terminated processes: {', '.join(terminated)}")
    else:
        print("[OK] No running Colab MCP processes found.")
    if cache_cleared:
        print("[OK] Cleared schema session directory.")
    print("[OK] Session reset. You can now connect using a different Google account.")


def run_launch(_args: argparse.Namespace) -> None:
    """Handle launch CLI command."""
    print("==> Launching Google Colab in browser...")
    success = ColabApplicationService.launch_browser()
    if success:
        print("[OK] Browser opened. Verify your account and select GPU/TPU in Runtime settings.")
    else:
        print("[X] Failed to open browser.")


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Colab MCP Management Tool (Domain-Driven Design Architecture)."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("setup", help="Install uv, cache repo, and configure mcp_config.json")
    subparsers.add_parser("status", help="Inspect environment, MCP registration, and running processes")
    subparsers.add_parser("reset", help="Kill existing colab processes and reset caches to switch Google accounts")
    subparsers.add_parser("launch", help="Open Google Colab in browser")

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "setup": run_setup,
        "status": run_status,
        "reset": run_reset,
        "launch": run_launch,
    }

    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
