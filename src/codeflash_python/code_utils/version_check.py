"""Version checking utilities for codeflash."""

from __future__ import annotations

import logging
import time

import requests
from packaging import version

logger = logging.getLogger("codeflash_python")

try:
    from codeflash_python.version import __version__
except ImportError:
    __version__: str = "0.0.0"

# Simple cache to avoid checking too frequently
_version_cache: dict[str, str | float] = {"version": "0.0.0", "timestamp": float(0)}
_cache_duration = 3600  # 1 hour cache


def get_latest_version_from_pypi() -> str | None:
    """Get the latest version of codeflash from PyPI.

    Returns:
        The latest version string from PyPI, or None if the request fails.

    """
    # Check cache first
    current_time = time.time()
    cached_version = _version_cache["version"]
    cached_timestamp = _version_cache["timestamp"]
    assert isinstance(cached_timestamp, float)
    if cached_version is not None and current_time - cached_timestamp < _cache_duration:
        assert isinstance(cached_version, str)
        return cached_version

    try:
        response = requests.get("https://pypi.org/pypi/codeflash/json", timeout=2)
        if response.status_code == 200:
            data = response.json()
            latest_version = data["info"]["version"]

            # Update cache
            _version_cache["version"] = latest_version
            _version_cache["timestamp"] = current_time

            return latest_version
        logger.debug("Failed to fetch version from PyPI: %s", response.status_code)
        return None
    except requests.RequestException as e:
        logger.debug("Network error fetching version from PyPI: %s", e)
        return None
    except (KeyError, ValueError) as e:
        logger.debug("Invalid response format from PyPI: %s", e)
        return None
    except Exception as e:
        logger.debug("Unexpected error fetching version from PyPI: %s", e)
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

    try:
        current_parsed = version.parse(__version__)
        latest_parsed = version.parse(latest_version)

        # Check if there's a newer minor version available
        # We only notify for minor version updates, not patch updates
        if latest_parsed > current_parsed:  # < > == operators can be directly applied on version objects
            logger.warning("A newer version(%s) of Codeflash is available, please update soon!", latest_version)

    except version.InvalidVersion as e:
        logger.debug("Invalid version format: %s", e)
        return
