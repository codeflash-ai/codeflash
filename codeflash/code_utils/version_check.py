"""Version checking utilities for codeflash."""

from __future__ import annotations

import time

import requests
from packaging import version

from codeflash.cli_cmds.console import logger
from codeflash.version import __version__

# Simple cache to avoid checking too frequently
_version_cache = {"version": "0.0.0", "timestamp": float(0)}
_cache_duration = 3600  # 1 hour cache


def get_latest_version_from_pypi() -> str | None:
    """Get the latest version of codeflash from PyPI.

    Returns:
        The latest version string from PyPI, or None if the request fails.

    """
    current_time = time.time()
    # Use local variables for fast access
    cache = _version_cache
    cache_duration = _cache_duration

    if cache["version"] is not None and current_time - cache["timestamp"] < cache_duration:
        return cache["version"]

    try:
        response = requests.get("https://pypi.org/pypi/codeflash/json", timeout=2)
        if response.ok:
            data = response.json()
            latest_version = data["info"]["version"]

            # Update cache
            cache["version"] = latest_version
            cache["timestamp"] = current_time

            return latest_version
        logger.debug(f"Failed to fetch version from PyPI: {response.status_code}")
        return None  # noqa: TRY300
    except requests.RequestException as e:
        logger.debug(f"Network error fetching version from PyPI: {e}")
        return None
    except (KeyError, ValueError) as e:
        logger.debug(f"Invalid response format from PyPI: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching version from PyPI: {e}")
        return None


def check_for_newer_minor_version() -> None:
    """Check if a newer minor version is available on PyPI and notify the user.

    This function compares the current version with the latest version on PyPI.
    If a newer minor version is available, it prints an informational message
    suggesting the user upgrade.
    """
    latest_version = get_latest_version_from_pypi()

    if not latest_version:
        return

    # Minimize attribute lookups
    version_parse = version.parse
    InvalidVersion = version.InvalidVersion

    try:
        current_parsed = version_parse(__version__)
        latest_parsed = version_parse(latest_version)

        # Check if there's a newer minor version available
        # We only notify for minor version updates, not patch updates
        if latest_parsed > current_parsed:
            logger.warning(f"A newer version({latest_version}) of Codeflash is available, please update soon!")

    except InvalidVersion as e:
        logger.debug(f"Invalid version format: {e}")
        return
