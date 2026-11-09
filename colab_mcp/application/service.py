"""Application Service Facade Module.

Provides a unified application service wrapping use-case invocations.
"""

from typing import List, Tuple

from colab_mcp.application.use_cases import (
    InspectStatusUseCase,
    LaunchColabUseCase,
    ResetSessionUseCase,
    SetupUseCase,
)
from colab_mcp.domain import SystemStatusReport


class ColabApplicationService:
    """Application service coordinating Colab MCP use cases."""

    @staticmethod
    def setup_environment() -> bool:
        """Run complete environment setup."""
        return SetupUseCase.execute()

    @staticmethod
    def reset_active_session() -> Tuple[List[str], bool]:
        """Reset running sessions and clear caches."""
        return ResetSessionUseCase.execute()

    @staticmethod
    def inspect_system_status() -> SystemStatusReport:
        """Retrieve current system status."""
        return InspectStatusUseCase.execute()

    @staticmethod
    def launch_browser() -> bool:
        """Open Colab in web browser."""
        return LaunchColabUseCase.execute()
