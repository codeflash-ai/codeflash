from __future__ import annotations

from pathlib import Path


def validate_java_module_resolution(source_file: Path, project_root: Path, module_root: Path) -> tuple[bool, str]:
    """Validate that a Java source file can be compiled and tested within the given module structure.

    Checks:
    - Source file exists
    - Source file is within project root
    - A build config (pom.xml, build.gradle, build.gradle.kts) exists in project root
    - Source file is within module root
    - Package declaration matches directory structure

    Returns:
        (True, "") if valid, (False, error_message) if invalid.

    """
    source_file = source_file.resolve()
    project_root = project_root.resolve()
    module_root = module_root.resolve()

    if not source_file.exists():
        return False, f"Source file does not exist: {source_file}"

    try:
        source_file.relative_to(project_root)
    except ValueError:
        return False, f"Source file {source_file} is outside the project root {project_root}"

    has_build_config = (
        (project_root / "pom.xml").exists()
        or (project_root / "build.gradle").exists()
        or (project_root / "build.gradle.kts").exists()
    )
    if not has_build_config:
        return False, f"No build configuration (pom.xml, build.gradle, build.gradle.kts) found in {project_root}"

    try:
        source_file.relative_to(module_root)
    except ValueError:
        return False, f"Source file {source_file} is outside the module root {module_root}"

    # Validate package declaration matches directory structure
    package_name = _parse_package_declaration(source_file)
    if package_name is not None:
        expected_dir = module_root / Path(*package_name.split("."))
        actual_dir = source_file.parent
        if actual_dir.resolve() != expected_dir.resolve():
            return False, (
                f"Package declaration '{package_name}' does not match directory structure. "
                f"Expected file at {expected_dir}, but found at {actual_dir}"
            )

    return True, ""


def _parse_package_declaration(source_file: Path) -> str | None:
    """Extract the package name from a Java source file, or None if no package declaration."""
    try:
        content = source_file.read_text(encoding="utf-8")
    except Exception:
        return None

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("package "):
            return stripped[8:].rstrip(";").strip()
        # Skip comments and blank lines at the top of the file
        if (
            stripped
            and not stripped.startswith("//")
            and not stripped.startswith("/*")
            and not stripped.startswith("*")
        ):
            break
    return None


def infer_java_module_root(source_file: Path, project_root: Path | None = None) -> Path:
    """Infer the correct Java module root (source root) for a source file.

    If project_root is None, walks up from source_file to find a build config.
    Then uses find_source_root() from build_tools, falling back to
    project_root/src/main/java, then project_root itself.
    """
    source_file = source_file.resolve()

    if project_root is None:
        project_root = _find_project_root_from_file(source_file)
    else:
        project_root = project_root.resolve()

    if project_root is None:
        return source_file.parent

    from codeflash.languages.java.build_tools import find_source_root

    source_root = find_source_root(project_root)
    if source_root is not None:
        return source_root

    # Fall back to standard Maven layout
    standard_src = project_root / "src" / "main" / "java"
    if standard_src.exists():
        return standard_src

    return project_root


def _find_project_root_from_file(source_file: Path) -> Path | None:
    """Walk up from source_file to find a directory with a build config."""
    current = source_file.parent
    while current != current.parent:
        if (current / "pom.xml").exists():
            return current
        if (current / "build.gradle").exists() or (current / "build.gradle.kts").exists():
            return current
        current = current.parent
    return None
