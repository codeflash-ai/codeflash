"""Java import resolution.

This module provides functionality to resolve Java imports to actual file paths
within a project, handling both source and test directories.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeflash.languages.java.build_tools import find_source_root, find_test_root, get_project_info
from codeflash.languages.java.parser import get_java_analyzer

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.languages.java.parser import JavaAnalyzer, JavaImportInfo

logger = logging.getLogger(__name__)


@dataclass
class ResolvedImport:
    """A resolved Java import."""

    import_path: str  # Original import path (e.g., "com.example.utils.StringUtils")
    file_path: Path | None  # Resolved file path, or None if external/unresolved
    is_external: bool  # True if this is an external dependency (not in project)
    is_wildcard: bool  # True if this was a wildcard import
    class_name: str | None  # The imported class name (e.g., "StringUtils")


class JavaImportResolver:
    """Resolves Java imports to file paths within a project."""

    # Standard Java packages that are always external
    STANDARD_PACKAGES = frozenset(["java", "javax", "sun", "com.sun", "jdk", "org.w3c", "org.xml", "org.ietf"])

    # Common third-party package prefixes
    COMMON_EXTERNAL_PREFIXES = frozenset(
        [
            "org.junit",
            "org.mockito",
            "org.assertj",
            "org.hamcrest",
            "org.slf4j",
            "org.apache",
            "org.springframework",
            "com.google",
            "com.fasterxml",
            "io.netty",
            "io.github",
            "lombok",
        ]
    )

    def __init__(self, project_root: Path) -> None:
        """Initialize the import resolver.

        Args:
            project_root: Root directory of the Java project.

        """
        self.project_root = project_root
        self._source_roots: list[Path] = []
        self._test_roots: list[Path] = []
        self._package_to_path_cache: dict[str, Path | None] = {}

        # Discover source and test roots
        self._discover_roots()

    def _discover_roots(self) -> None:
        """Discover source and test root directories."""
        # Try to get project info first
        project_info = get_project_info(self.project_root)

        if project_info:
            self._source_roots = project_info.source_roots
            self._test_roots = project_info.test_roots
        else:
            # Fall back to standard detection
            source_root = find_source_root(self.project_root)
            if source_root:
                self._source_roots = [source_root]

            test_root = find_test_root(self.project_root)
            if test_root:
                self._test_roots = [test_root]

    def resolve_import(self, import_info: JavaImportInfo) -> ResolvedImport:
        """Resolve a single import to a file path.

        Args:
            import_info: The import to resolve.

        Returns:
            ResolvedImport with resolution details.

        """
        import_path = import_info.import_path

        # Check if it's a standard library import
        if self._is_standard_library(import_path):
            return ResolvedImport(
                import_path=import_path,
                file_path=None,
                is_external=True,
                is_wildcard=import_info.is_wildcard,
                class_name=self._extract_class_name(import_path),
            )

        # Check if it's a known external library
        if self._is_external_library(import_path):
            return ResolvedImport(
                import_path=import_path,
                file_path=None,
                is_external=True,
                is_wildcard=import_info.is_wildcard,
                class_name=self._extract_class_name(import_path),
            )

        # Try to resolve within the project
        resolved_path = self._resolve_to_file(import_path)

        return ResolvedImport(
            import_path=import_path,
            file_path=resolved_path,
            is_external=resolved_path is None,
            is_wildcard=import_info.is_wildcard,
            class_name=self._extract_class_name(import_path),
        )

    def resolve_imports(self, imports: list[JavaImportInfo]) -> list[ResolvedImport]:
        """Resolve multiple imports.

        Args:
            imports: List of imports to resolve.

        Returns:
            List of ResolvedImport objects.

        """
        return [self.resolve_import(imp) for imp in imports]

    def _is_standard_library(self, import_path: str) -> bool:
        """Check if an import is from the Java standard library."""
        return any(import_path.startswith(prefix + ".") or import_path == prefix for prefix in self.STANDARD_PACKAGES)

    def _is_external_library(self, import_path: str) -> bool:
        """Check if an import is from a known external library."""
        for prefix in self.COMMON_EXTERNAL_PREFIXES:
            if import_path.startswith(prefix + ".") or import_path == prefix:
                return True
        return False

    def _resolve_to_file(self, import_path: str) -> Path | None:
        """Try to resolve an import path to a file in the project.

        Args:
            import_path: The fully qualified import path.

        Returns:
            Path to the Java file, or None if not found.

        """
        # Check cache
        if import_path in self._package_to_path_cache:
            return self._package_to_path_cache[import_path]

        # Convert package path to file path
        # e.g., "com.example.utils.StringUtils" -> "com/example/utils/StringUtils.java"
        relative_path = import_path.replace(".", "/") + ".java"

        # Search in source roots
        for source_root in self._source_roots:
            candidate = source_root / relative_path
            if candidate.exists():
                self._package_to_path_cache[import_path] = candidate
                return candidate

        # Search in test roots
        for test_root in self._test_roots:
            candidate = test_root / relative_path
            if candidate.exists():
                self._package_to_path_cache[import_path] = candidate
                return candidate

        # Not found
        self._package_to_path_cache[import_path] = None
        return None

    def _extract_class_name(self, import_path: str) -> str | None:
        """Extract the class name from an import path.

        Args:
            import_path: The import path (e.g., "com.example.MyClass").

        Returns:
            The class name (e.g., "MyClass"), or None if it's a wildcard.

        """
        if not import_path:
            return None
        # Use rpartition to avoid allocating a list from split()
        last_part = import_path.rpartition(".")[2]
        if last_part and last_part[0].isupper():
            return last_part
        return None

    def find_class_file(self, class_name: str, package_hint: str | None = None) -> Path | None:
        """Find the file containing a specific class.

        Args:
            class_name: The simple class name (e.g., "StringUtils").
            package_hint: Optional package hint to narrow the search.

        Returns:
            Path to the Java file, or None if not found.

        """
        if package_hint:
            # Try the exact path first
            import_path = f"{package_hint}.{class_name}"
            result = self._resolve_to_file(import_path)
            if result:
                return result

        # Search all source and test roots for the class
        file_name = f"{class_name}.java"

        for root in self._source_roots + self._test_roots:
            for java_file in root.rglob(file_name):
                return java_file

        return None

    def get_imports_from_file(self, file_path: Path, analyzer: JavaAnalyzer | None = None) -> list[ResolvedImport]:
        """Get and resolve all imports from a Java file.

        Args:
            file_path: Path to the Java file.
            analyzer: Optional JavaAnalyzer instance.

        Returns:
            List of ResolvedImport objects.

        """
        analyzer = analyzer or get_java_analyzer()

        try:
            source = file_path.read_text(encoding="utf-8")
            imports = analyzer.find_imports(source)
            return self.resolve_imports(imports)
        except Exception as e:
            logger.warning("Failed to get imports from %s: %s", file_path, e)
            return []

    def get_project_imports(self, file_path: Path, analyzer: JavaAnalyzer | None = None) -> list[ResolvedImport]:
        """Get only the imports that resolve to files within the project.

        Args:
            file_path: Path to the Java file.
            analyzer: Optional JavaAnalyzer instance.

        Returns:
            List of ResolvedImport objects for project-internal imports only.

        """
        all_imports = self.get_imports_from_file(file_path, analyzer)
        return [imp for imp in all_imports if not imp.is_external and imp.file_path is not None]


def resolve_imports_for_file(
    file_path: Path, project_root: Path, analyzer: JavaAnalyzer | None = None
) -> list[ResolvedImport]:
    """Convenience function to resolve imports for a single file.

    Args:
        file_path: Path to the Java file.
        project_root: Root directory of the project.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        List of ResolvedImport objects.

    """
    resolver = JavaImportResolver(project_root)
    return resolver.get_imports_from_file(file_path, analyzer)


def find_helper_files(
    file_path: Path, project_root: Path, max_depth: int = 2, analyzer: JavaAnalyzer | None = None
) -> dict[Path, list[str]]:
    """Find helper files imported by a Java file, recursively.

    This traces the import chain to find all project files that the
    given file depends on, up to max_depth levels.

    Args:
        file_path: Path to the Java file.
        project_root: Root directory of the project.
        max_depth: Maximum depth of import chain to follow.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Dict mapping file paths to list of imported class names.

    """
    resolver = JavaImportResolver(project_root)
    analyzer = analyzer or get_java_analyzer()

    result: dict[Path, list[str]] = {}
    visited: set[Path] = {file_path}

    def _trace_imports(current_file: Path, depth: int) -> None:
        if depth > max_depth:
            return

        project_imports = resolver.get_project_imports(current_file, analyzer)

        for imp in project_imports:
            if imp.file_path and imp.file_path not in visited:
                visited.add(imp.file_path)

                if imp.file_path not in result:
                    result[imp.file_path] = []

                if imp.class_name:
                    result[imp.file_path].append(imp.class_name)

                # Recurse into the imported file
                _trace_imports(imp.file_path, depth + 1)

    _trace_imports(file_path, 0)

    return result
