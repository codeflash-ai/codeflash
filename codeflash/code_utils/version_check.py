"""Version checking utilities for codeflash."""

import sys
import time
from typing import Optional

import requests
from packaging import version

from codeflash.cli_cmds.console import console, logger
from codeflash.version import __version__

# Simple cache to avoid checking too frequently
_version_cache = {"version": None, "timestamp": 0}
_cache_duration = 3600  # 1 hour cache


def get_latest_version_from_pypi() -> Optional[str]:
    """Get the latest version of codeflash from PyPI.
    
    Returns:
        The latest version string from PyPI, or None if the request fails.
    """
    global _version_cache
    
    # Check cache first
    current_time = time.time()
    if (_version_cache["version"] is not None and 
        current_time - _version_cache["timestamp"] < _cache_duration):
        return _version_cache["version"]
    
    try:
        response = requests.get("https://pypi.org/pypi/codeflash/json", timeout=10)
        if response.status_code == 200:
            data = response.json()
            latest_version = data["info"]["version"]
            
            # Update cache
            _version_cache["version"] = latest_version
            _version_cache["timestamp"] = current_time
            
            return latest_version
        else:
            logger.debug(f"Failed to fetch version from PyPI: {response.status_code}")
            return None
    except requests.RequestException as e:
        logger.debug(f"Network error fetching version from PyPI: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.debug(f"Invalid response format from PyPI: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching version from PyPI: {e}")
        return None


def check_for_newer_minor_version(*, disable_check: bool = False) -> None:
    """Check if a newer minor version is available on PyPI and notify the user.
    
    This function compares the current version with the latest version on PyPI.
    If a newer minor version is available, it prints an informational message
    suggesting the user upgrade.
    
    Args:
        disable_check: If True, skip the version check entirely.
    """
    if disable_check:
        return
        
    # Get current version dynamically to handle runtime changes
    from codeflash.version import __version__ as current_version
    latest_version = get_latest_version_from_pypi()
    
    if not latest_version:
        return
    
    try:
        current_parsed = version.parse(current_version)
        latest_parsed = version.parse(latest_version)
        
        # Check if there's a newer minor version available
        # We only notify for minor version updates, not patch updates
        if (latest_parsed.major > current_parsed.major or 
            (latest_parsed.major == current_parsed.major and 
             latest_parsed.minor > current_parsed.minor)):
            
            console.print(
                f"[bold blue]ℹ️  A newer version of Codeflash is available![/bold blue]\n"
                f"Current version: {current_version} | Latest version: {latest_version}\n"
                f"Consider upgrading for better quality optimizations.",
                style="blue"
            )
            
    except version.InvalidVersion as e:
        logger.debug(f"Invalid version format: {e}")
        return 