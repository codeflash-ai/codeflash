"""Java generated test postprocessing helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from codeflash.models.models import GeneratedTestsList


def get_java_sources_root(tests_root: Path) -> Path:
    """Get the Java sources root directory for test files.

    For Java projects, tests_root might include the package path
    (e.g., test/src/com/aerospike/test). We need to find the base directory
    that should contain the package directories, not the tests_root itself.

    This method looks for standard Java package prefixes (com, org, net, io, edu, gov)
    in the tests_root path and returns everything before that prefix.

    Returns:
        Path to the Java sources root directory.

    """
    parts = tests_root.parts

    # Check if tests_root already ends with "src" (already a Java sources root)
    if tests_root.name == "src":
        logger.debug(f"[JAVA] tests_root already ends with 'src': {tests_root}")
        logger.debug(f"[JAVA-ROOT] Returning Java sources root: {tests_root}, tests_root was: {tests_root}")
        return tests_root

    # Check if tests_root already ends with src/test/java (Maven-standard)
    if len(parts) >= 3 and parts[-3:] == ("src", "test", "java"):
        logger.debug(f"[JAVA] tests_root already is Maven-standard test directory: {tests_root}")
        logger.debug(f"[JAVA-ROOT] Returning Java sources root: {tests_root}, tests_root was: {tests_root}")
        return tests_root

    # Check for simple "src" subdirectory (handles test/src, test-module/src, etc.)
    src_subdir = tests_root / "src"
    if src_subdir.exists() and src_subdir.is_dir():
        logger.debug(f"[JAVA] Found 'src' subdirectory: {src_subdir}")
        logger.debug(f"[JAVA-ROOT] Returning Java sources root: {src_subdir}, tests_root was: {tests_root}")
        return src_subdir

    # Check for Maven-standard src/test/java structure as subdirectory
    maven_test_dir = tests_root / "src" / "test" / "java"
    if maven_test_dir.exists() and maven_test_dir.is_dir():
        logger.debug(f"[JAVA] Found Maven-standard test directory as subdirectory: {maven_test_dir}")
        logger.debug(f"[JAVA-ROOT] Returning Java sources root: {maven_test_dir}, tests_root was: {tests_root}")
        return maven_test_dir

    # Look for standard Java package prefixes that indicate the start of package structure
    standard_package_prefixes = ("com", "org", "net", "io", "edu", "gov")

    for i, part in enumerate(parts):
        if part in standard_package_prefixes:
            # Found start of package path, return everything before it
            if i > 0:
                java_sources_root = Path(*parts[:i])
                logger.debug(f"[JAVA] Detected Java sources root: {java_sources_root} (from tests_root: {tests_root})")
                logger.debug(
                    f"[JAVA-ROOT] Returning Java sources root: {java_sources_root}, tests_root was: {tests_root}"
                )
                return java_sources_root

    # If no standard package prefix found, check if there's a 'java' directory
    # (standard Maven structure: src/test/java)
    for i, part in enumerate(parts):
        if part == "java" and i > 0:
            # Return up to and including 'java'
            java_sources_root = Path(*parts[: i + 1])
            logger.debug(f"[JAVA] Detected Maven-style Java sources root: {java_sources_root}")
            logger.debug(
                f"[JAVA-ROOT] Returning Java sources root: {java_sources_root}, tests_root was: {tests_root}"
            )
            return java_sources_root

    # Default: return tests_root as-is (original behavior)
    logger.debug(f"[JAVA] Using tests_root as Java sources root: {tests_root}")
    logger.debug(f"[JAVA-ROOT] Returning Java sources root: {tests_root}, tests_root was: {tests_root}")
    return tests_root


def fix_java_test_paths(
    behavior_source: str, perf_source: str, used_paths: set[Path], tests_root: Path
) -> tuple[Path, Path, str, str]:
    """Fix Java test file paths to match package structure.

    Java requires test files to be in directories matching their package.
    This method extracts the package and class from the generated tests
    and returns correct paths. If the path would conflict with an already
    used path, it renames the class by adding an index suffix.

    Args:
        behavior_source: Source code of the behavior test.
        perf_source: Source code of the performance test.
        used_paths: Set of already used behavior file paths.
        tests_root: Root directory for tests.

    Returns:
        Tuple of (behavior_path, perf_path, modified_behavior_source, modified_perf_source)
        with correct package structure and unique class names.

    """
    # Extract package from behavior source
    package_match = re.search(r"^\s*package\s+([\w.]+)\s*;", behavior_source, re.MULTILINE)
    package_name = package_match.group(1) if package_match else ""

    # JPMS: If a test module-info.java exists, remap the package to the
    # test module namespace to avoid split-package errors.
    # E.g., io.questdb.cairo -> io.questdb.test.cairo
    test_dir = get_java_sources_root(tests_root)
    test_module_info = test_dir / "module-info.java"
    if package_name and test_module_info.exists():
        mi_content = test_module_info.read_text()
        mi_match = re.search(r"module\s+([\w.]+)", mi_content)
        if mi_match:
            test_module_name = mi_match.group(1)
            main_dir = test_dir.parent.parent / "main" / "java"
            main_module_info = main_dir / "module-info.java"
            if main_module_info.exists():
                main_content = main_module_info.read_text()
                main_match = re.search(r"module\s+([\w.]+)", main_content)
                if main_match:
                    main_module_name = main_match.group(1)
                    if package_name.startswith(main_module_name):
                        suffix = package_name[len(main_module_name) :]
                        new_package = test_module_name + suffix
                        old_decl = f"package {package_name};"
                        new_decl = f"package {new_package};"
                        behavior_source = behavior_source.replace(old_decl, new_decl, 1)
                        perf_source = perf_source.replace(old_decl, new_decl, 1)
                        package_name = new_package
                        logger.debug(f"[JPMS] Remapped package: {old_decl} -> {new_decl}")

    # Extract class name from behavior source
    # Use more specific pattern to avoid matching words like "command" or text in comments
    class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", behavior_source, re.MULTILINE)
    behavior_class = class_match.group(1) if class_match else "GeneratedTest"

    # Extract class name from perf source
    perf_class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", perf_source, re.MULTILINE)
    perf_class = perf_class_match.group(1) if perf_class_match else "GeneratedPerfTest"

    # Build paths with package structure
    # Use the Java sources root, not tests_root, to avoid path duplication
    # when tests_root already includes the package path
    test_dir = get_java_sources_root(tests_root)

    if package_name:
        package_path = package_name.replace(".", "/")
        behavior_path = test_dir / package_path / f"{behavior_class}.java"
        perf_path = test_dir / package_path / f"{perf_class}.java"
    else:
        package_path = ""
        behavior_path = test_dir / f"{behavior_class}.java"
        perf_path = test_dir / f"{perf_class}.java"

    # If path already used, rename class by adding index suffix
    modified_behavior_source = behavior_source
    modified_perf_source = perf_source
    if behavior_path in used_paths:
        # Find a unique index
        index = 2
        while True:
            new_behavior_class = f"{behavior_class}_{index}"
            new_perf_class = f"{perf_class}_{index}"
            if package_path:
                new_behavior_path = test_dir / package_path / f"{new_behavior_class}.java"
                new_perf_path = test_dir / package_path / f"{new_perf_class}.java"
            else:
                new_behavior_path = test_dir / f"{new_behavior_class}.java"
                new_perf_path = test_dir / f"{new_perf_class}.java"
            if new_behavior_path not in used_paths:
                behavior_path = new_behavior_path
                perf_path = new_perf_path
                # Rename class in source code - replace the class declaration
                modified_behavior_source = re.sub(
                    rf"^((?:public\s+)?class\s+){re.escape(behavior_class)}(\b)",
                    rf"\g<1>{new_behavior_class}\g<2>",
                    behavior_source,
                    count=1,
                    flags=re.MULTILINE,
                )
                modified_perf_source = re.sub(
                    rf"^((?:public\s+)?class\s+){re.escape(perf_class)}(\b)",
                    rf"\g<1>{new_perf_class}\g<2>",
                    perf_source,
                    count=1,
                    flags=re.MULTILINE,
                )
                logger.debug(f"[JAVA] Renamed duplicate test class from {behavior_class} to {new_behavior_class}")
                break
            index += 1

    # Create directories if needed
    behavior_path.parent.mkdir(parents=True, exist_ok=True)
    perf_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"[JAVA] Fixed paths: behavior={behavior_path}, perf={perf_path}")
    logger.debug(
        f"[WRITE-PATH] Writing test to behavior_path={behavior_path}, perf_path={perf_path}, "
        f"package={package_name}, behavior_class={behavior_class}, perf_class={perf_class}"
    )
    return behavior_path, perf_path, modified_behavior_source, modified_perf_source


def postprocess_generated_tests(generated_tests: GeneratedTestsList, tests_root: Path) -> GeneratedTestsList:
    """Postprocess generated Java tests to ensure package-aligned paths."""
    used_behavior_paths: set[Path] = set()
    for generated_test in generated_tests.generated_tests:
        behavior_path, perf_path, modified_behavior_source, modified_perf_source = fix_java_test_paths(
            generated_test.instrumented_behavior_test_source,
            generated_test.instrumented_perf_test_source,
            used_behavior_paths,
            tests_root,
        )
        generated_test.behavior_file_path = behavior_path
        generated_test.perf_file_path = perf_path
        generated_test.instrumented_behavior_test_source = modified_behavior_source
        generated_test.instrumented_perf_test_source = modified_perf_source
        used_behavior_paths.add(behavior_path)
    return generated_tests


def fix_existing_test_class_names(
    injected_behavior_test: str | None, injected_perf_test: str | None
) -> tuple[str | None, str | None]:
    """Fix Java class names for existing instrumented tests."""
    if injected_behavior_test is not None:
        injected_behavior_test = injected_behavior_test.replace("__perfinstrumented", "__existing_perfinstrumented")
    if injected_perf_test is not None:
        injected_perf_test = injected_perf_test.replace("__perfonlyinstrumented", "__existing_perfonlyinstrumented")
    return injected_behavior_test, injected_perf_test
