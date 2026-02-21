"""Java code formatting.

This module provides functionality to format Java code using
google-java-format or other available formatters.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_FORMATTER_CACHE: dict[str | None, JavaFormatter] = {}

logger = logging.getLogger(__name__)


class JavaFormatter:
    """Java code formatter using google-java-format or fallback methods."""

    # Path to google-java-format JAR (if downloaded)
    _google_java_format_jar: Path | None = None

    # Version of google-java-format to use
    GOOGLE_JAVA_FORMAT_VERSION = "1.19.2"

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize the Java formatter.

        Args:
            project_root: Optional project root for project-specific formatting rules.

        """
        self.project_root = project_root
        self._java_executable = self._find_java()

    def _find_java(self) -> str | None:
        """Find the Java executable."""
        # Check JAVA_HOME
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            java_path = Path(java_home) / "bin" / "java"
            if java_path.exists():
                return str(java_path)

        # Check PATH
        java_path = shutil.which("java")
        if java_path:
            return java_path

        return None

    def format_code(self, source: str, file_path: Path | None = None) -> str:
        """Format Java source code.

        Attempts to use google-java-format if available, otherwise
        returns the source unchanged.

        Args:
            source: The Java source code to format.
            file_path: Optional file path for context.

        Returns:
            Formatted source code.

        """
        if not source or not source.strip():
            return source

        # Try google-java-format first
        formatted = self._format_with_google_java_format(source)
        if formatted is not None:
            return formatted

        # Try Eclipse formatter (if available in project)
        if self.project_root:
            formatted = self._format_with_eclipse(source)
            if formatted is not None:
                return formatted

        # Return original source if no formatter available
        logger.debug("No Java formatter available, returning original source")
        return source

    def _format_with_google_java_format(self, source: str) -> str | None:
        """Format using google-java-format.

        Args:
            source: The source code to format.

        Returns:
            Formatted source, or None if formatting failed.

        """
        if not self._java_executable:
            return None

        # Try to find or download google-java-format
        jar_path = self._get_google_java_format_jar()
        if not jar_path:
            return None

        try:
            # Write source to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False, encoding="utf-8") as tmp:
                tmp.write(source)
                tmp_path = tmp.name

            try:
                result = subprocess.run(
                    [self._java_executable, "-jar", str(jar_path), "--replace", tmp_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    # Read back the formatted file
                    with Path(tmp_path).open(encoding="utf-8") as f:
                        return f.read()
                else:
                    logger.debug("google-java-format failed: %s", result.stderr or result.stdout)

            finally:
                # Clean up temp file
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()

        except subprocess.TimeoutExpired:
            logger.warning("google-java-format timed out")
        except Exception as e:
            logger.debug("google-java-format error: %s", e)

        return None

    def _get_google_java_format_jar(self) -> Path | None:
        """Get path to google-java-format JAR, downloading if necessary.

        Returns:
            Path to the JAR file, or None if not available.

        """
        if JavaFormatter._google_java_format_jar:
            if JavaFormatter._google_java_format_jar.exists():
                return JavaFormatter._google_java_format_jar

        # Check common locations
        possible_paths = [
            # In project's .codeflash directory
            self.project_root / ".codeflash" / f"google-java-format-{self.GOOGLE_JAVA_FORMAT_VERSION}-all-deps.jar"
            if self.project_root
            else None,
            # In user's home directory
            Path.home() / ".codeflash" / f"google-java-format-{self.GOOGLE_JAVA_FORMAT_VERSION}-all-deps.jar",
            # In system temp
            Path(tempfile.gettempdir())
            / "codeflash"
            / f"google-java-format-{self.GOOGLE_JAVA_FORMAT_VERSION}-all-deps.jar",
        ]

        for path in possible_paths:
            if path and path.exists():
                JavaFormatter._google_java_format_jar = path
                return path

        # Don't auto-download to avoid surprises
        # Users can manually download the JAR
        logger.debug(
            "google-java-format JAR not found. Download from https://github.com/google/google-java-format/releases"
        )
        return None

    def _format_with_eclipse(self, source: str) -> str | None:
        """Format using Eclipse formatter settings (if available in project).

        Args:
            source: The source code to format.

        Returns:
            Formatted source, or None if formatting failed.

        """
        # Eclipse formatter requires eclipse.ini or a config file
        # This is a placeholder for future implementation
        return None

    def download_google_java_format(self, target_dir: Path | None = None) -> Path | None:
        """Download google-java-format JAR.

        Args:
            target_dir: Directory to download to (defaults to ~/.codeflash/).

        Returns:
            Path to the downloaded JAR, or None if download failed.

        """
        import urllib.request

        target_dir = target_dir or Path.home() / ".codeflash"
        target_dir.mkdir(parents=True, exist_ok=True)

        jar_name = f"google-java-format-{self.GOOGLE_JAVA_FORMAT_VERSION}-all-deps.jar"
        jar_path = target_dir / jar_name

        if jar_path.exists():
            JavaFormatter._google_java_format_jar = jar_path
            return jar_path

        url = (
            f"https://github.com/google/google-java-format/releases/download/"
            f"v{self.GOOGLE_JAVA_FORMAT_VERSION}/{jar_name}"
        )

        try:
            logger.info("Downloading google-java-format from %s", url)
            urllib.request.urlretrieve(url, jar_path)  # noqa: S310
            JavaFormatter._google_java_format_jar = jar_path
            logger.info("Downloaded google-java-format to %s", jar_path)
            return jar_path
        except Exception as e:
            logger.exception("Failed to download google-java-format: %s", e)
            return None


def format_java_code(source: str, project_root: Path | None = None) -> str:
    """Convenience function to format Java code.

    Args:
        source: The Java source code to format.
        project_root: Optional project root for context.

    Returns:
        Formatted source code.

    """
    formatter = _get_cached_formatter(project_root)
    return formatter.format_code(source)


def format_java_file(file_path: Path, in_place: bool = False) -> str:
    """Format a Java file.

    Args:
        file_path: Path to the Java file.
        in_place: Whether to modify the file in place.

    Returns:
        Formatted source code.

    """
    source = file_path.read_text(encoding="utf-8")
    formatter = JavaFormatter(file_path.parent)
    formatted = formatter.format_code(source, file_path)

    if in_place and formatted != source:
        file_path.write_text(formatted, encoding="utf-8")

    return formatted


def normalize_java_code(source: str) -> str:
    """Normalize Java code for deduplication.

    This removes comments and normalizes whitespace to allow
    comparison of semantically equivalent code.

    Args:
        source: The Java source code.

    Returns:
        Normalized source code.

    """
    lines = source.splitlines()
    normalized_lines = []
    in_block_comment = False

    for line in lines:
        # Handle block comments
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
                line = line[line.index("*/") + 2 :]
            else:
                continue

        # Remove line comments
        if "//" in line:
            # Find // that's not inside a string
            in_string = False
            escape_next = False
            comment_start = -1
            for i, char in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not in_string:
                    in_string = True
                elif char == '"' and in_string:
                    in_string = False
                elif not in_string and i < len(line) - 1 and line[i : i + 2] == "//":
                    comment_start = i
                    break
            if comment_start >= 0:
                line = line[:comment_start]

        # Handle start of block comments
        if "/*" in line:
            start_idx = line.index("/*")
            if "*/" in line[start_idx:]:
                # Block comment on single line
                end_idx = line.index("*/", start_idx)
                line = line[:start_idx] + line[end_idx + 2 :]
            else:
                in_block_comment = True
                line = line[:start_idx]

        # Skip empty lines and add non-empty ones
        stripped = line.strip()
        if stripped:
            normalized_lines.append(stripped)

    return "\n".join(normalized_lines)


def _get_cached_formatter(project_root: Path | None) -> JavaFormatter:
    key = str(project_root) if project_root is not None else None
    fmt = _FORMATTER_CACHE.get(key)
    if fmt is None:
        fmt = JavaFormatter(project_root)
        _FORMATTER_CACHE[key] = fmt
    return fmt
