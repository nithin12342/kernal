"""Infrastructure Browser Adapter Module.

Handles launching web browsers across platforms with priority for system Chrome.
"""

import logging
import os
import subprocess
import webbrowser

logger = logging.getLogger("colab_mcp.infrastructure.browser")


class BrowserAdapter:
    """Interacts with system web browsers."""

    CHROME_DEFAULT_WINDOWS_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    @classmethod
    def open_url(cls, url: str) -> bool:
        """Launch the target URL using Chrome if available, otherwise default browser."""
        try:
            if os.name == "nt" and os.path.exists(cls.CHROME_DEFAULT_WINDOWS_PATH):
                subprocess.Popen([cls.CHROME_DEFAULT_WINDOWS_PATH, url])
                return True
            return webbrowser.open(url)
        except Exception as exc:
            logger.error(f"Failed to open browser URL {url}: {exc}")
            return False
