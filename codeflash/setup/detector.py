"""Universal project detection engine for Codeflash.

This module provides a single detection engine that works for all supported languages,
consolidating detection logic from various parts of the codebase.

Usage:
    from codeflash.setup import detect_project

    detected = detect_project()
    print(f"Language: {detected.language}")
    print(f"Module root: {detected.module_root}")
    print(f"Test runner: {detected.test_runner}")
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

_BUILD_DIRS = frozenset({"build", "dist", "out", ".next", ".nuxt"})


@dataclass
class DetectedProject:
    """Result of project auto-detection.

    All paths are absolute. The confidence score indicates how certain
    we are about the detection (0.0 = guessing, 1.0 = certain).
    """

    # Core detection results
    language: str  # "python" | "javascript" | "typescript" | "java"
    project_root: Path
    module_root: Path
    tests_root: Path | None

    # Tooling detection
    test_runner: str  # "pytest" | "jest" | "vitest" | "mocha"
    formatter_cmds: list[str]

    # Ignore paths (absolute paths to ignore)
    ignore_paths: list[Path] = field(default_factory=list)

    # Confidence score for the detection (0.0 - 1.0)
    confidence: float = 0.8

    # Detection details (for debugging/display)
    detection_details: dict[str, str] = field(default_factory=dict)

    def to_display_dict(self) -> dict[str, str]:
        """Convert to dictionary for display purposes."""
        formatter_display = self.formatter_cmds[0] if self.formatter_cmds else "none detected"
        if len(self.formatter_cmds) > 1:
            formatter_display += f" (+{len(self.formatter_cmds) - 1} more)"

        ignore_display = ", ".join(p.name for p in self.ignore_paths[:3])
        if len(self.ignore_paths) > 3:
            ignore_display += f" (+{len(self.ignore_paths) - 3} more)"

        return {
            "Language": self.language.capitalize(),
            "Module root": str(self.module_root.relative_to(self.project_root))
            if self.module_root != self.project_root
            else ".",
            "Tests root": str(self.tests_root.relative_to(self.project_root)) if self.tests_root else "not detected",
            "Test runner": self.test_runner,
            "Formatter": formatter_display or "none",
            "Ignoring": ignore_display or "defaults only",
        }


def detect_project(path: Path | None = None) -> DetectedProject:
    """Auto-detect all project settings.

    This is the main entry point for project detection. It finds the project root,
    detects the language, and auto-detects all configuration values.

    Args:
        path: Starting path for detection. Defaults to current working directory.

    Returns:
        DetectedProject with all detected settings.

    Raises:
        ValueError: If no valid project can be detected.

    """
    start_path = path or Path.cwd()
    detection_details: dict[str, str] = {}

    # Step 1: Find project root
    project_root = _find_project_root(start_path)
    if project_root is None:
        # No project root found, use start_path
        project_root = start_path
        detection_details["project_root"] = "using current directory (no markers found)"
    else:
        detection_details["project_root"] = f"found at {project_root}"

    # Step 2: Detect language
    language, lang_confidence, lang_detail = _detect_language(project_root)
    detection_details["language"] = lang_detail

    # Step 3: Detect module root
    module_root, module_detail = _detect_module_root(project_root, language)
    detection_details["module_root"] = module_detail

    # Step 4: Detect tests root
    tests_root, tests_detail = _detect_tests_root(project_root, language)
    detection_details["tests_root"] = tests_detail

    # Step 5: Detect test runner
    test_runner, runner_detail = _detect_test_runner(project_root, language)
    detection_details["test_runner"] = runner_detail

    # Step 6: Detect formatter
    formatter_cmds, formatter_detail = _detect_formatter(project_root, language)
    detection_details["formatter"] = formatter_detail

    # Step 7: Detect ignore paths
    ignore_paths, ignore_detail = _detect_ignore_paths(project_root, language)
    detection_details["ignore_paths"] = ignore_detail

    # Calculate overall confidence
    confidence = lang_confidence * 0.4 + 0.6  # Language detection is 40% of confidence

    return DetectedProject(
        language=language,
        project_root=project_root,
        module_root=module_root,
        tests_root=tests_root,
        test_runner=test_runner,
        formatter_cmds=formatter_cmds,
        ignore_paths=ignore_paths,
        confidence=confidence,
        detection_details=detection_details,
    )


def _find_project_root(start_path: Path) -> Path | None:
    """Find the project root by walking up the directory tree.

    Looks for:
    - .git directory (git repository root)
    - pyproject.toml (Python project)
    - package.json (JavaScript/TypeScript project)
    - Cargo.toml (Rust project - future)

    Args:
        start_path: Starting directory for search.

    Returns:
        Path to project root, or None if not found.

    """
    current = start_path.resolve()

    while current != current.parent:
        # Check for project markers
        markers = [".git", "pyproject.toml", "package.json", "Cargo.toml", "pom.xml", "build.gradle", "build.gradle.kts"]
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    return None


def _detect_language(project_root: Path) -> tuple[str, float, str]:
    """Detect the primary programming language of the project.

    Detection priority:
    1. tsconfig.json → TypeScript (high confidence)
    2. pyproject.toml or setup.py → Python (high confidence)
    3. package.json → JavaScript (medium confidence)
    4. File extension counting → best guess (low confidence)

    Args:
        project_root: Root directory of the project.

    Returns:
        Tuple of (language, confidence, detail_string).

    """
    has_tsconfig = (project_root / "tsconfig.json").exists()
    has_pyproject = (project_root / "pyproject.toml").exists()
    has_setup_py = (project_root / "setup.py").exists()
    has_package_json = (project_root / "package.json").exists()
    has_pom_xml = (project_root / "pom.xml").exists()
    has_build_gradle = (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists()

    # Java (pom.xml or build.gradle is definitive)
    if has_pom_xml:
        return "java", 1.0, "pom.xml found"
    if has_build_gradle:
        return "java", 1.0, "build.gradle found"

    # TypeScript (tsconfig.json is definitive)
    if has_tsconfig:
        return "typescript", 1.0, "tsconfig.json found"

    # Python (pyproject.toml or setup.py)
    if has_pyproject or has_setup_py:
        marker = "pyproject.toml" if has_pyproject else "setup.py"
        # Check if it's also a JS project (monorepo)
        if has_package_json:
            # Count files to determine primary language
            py_count = len(list(project_root.rglob("*.py")))
            js_count = len(list(project_root.rglob("*.js"))) + len(list(project_root.rglob("*.ts")))
            if js_count > py_count * 2:  # JS files significantly outnumber Python
                return "javascript", 0.7, "package.json found (more JS files than Python)"
        return "python", 1.0, f"{marker} found"

    # JavaScript (package.json without Python markers)
    if has_package_json:
        return "javascript", 0.9, "package.json found"

    # Fall back to file extension counting
    py_count = len(list(project_root.rglob("*.py")))
    js_count = len(list(project_root.rglob("*.js")))
    ts_count = len(list(project_root.rglob("*.ts")))
    java_count = len(list(project_root.rglob("*.java")))

    if java_count > 0 and java_count >= max(py_count, js_count, ts_count):
        return "java", 0.5, f"found {java_count} .java files"
    if ts_count > 0:
        return "typescript", 0.5, f"found {ts_count} .ts files"
    if js_count > py_count:
        return "javascript", 0.5, f"found {js_count} .js files"
    if py_count > 0:
        return "python", 0.5, f"found {py_count} .py files"

    # Default to Python
    return "python", 0.3, "defaulting to Python"


def _detect_module_root(project_root: Path, language: str) -> tuple[Path, str]:
    """Detect the module/source root directory.

    Args:
        project_root: Root directory of the project.
        language: Detected language.

    Returns:
        Tuple of (module_root_path, detail_string).

    """
    if language in ("javascript", "typescript"):
        return _detect_js_module_root(project_root)
    if language == "java":
        return _detect_java_module_root(project_root)
    return _detect_python_module_root(project_root)


def _detect_python_module_root(project_root: Path) -> tuple[Path, str]:
    """Detect Python module root.

    Priority:
    1. pyproject.toml [tool.poetry.name] or [project.name]
    2. src/ directory with __init__.py
    3. Directory with __init__.py matching project name
    4. src/ directory (even without __init__.py)
    5. Project root

    """
    # Try to get project name from pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    project_name = None

    if pyproject_path.exists():
        try:
            with pyproject_path.open("rb") as f:
                data = tomlkit.parse(f.read())

            # Try poetry name
            project_name = data.get("tool", {}).get("poetry", {}).get("name")
            # Try standard project name
            if not project_name:
                project_name = data.get("project", {}).get("name")
        except Exception:
            pass

    # Check for src layout
    src_dir = project_root / "src"
    if src_dir.is_dir():
        # Check for package inside src
        if project_name:
            pkg_dir = src_dir / project_name
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                return pkg_dir, f"src/{project_name}/ (from pyproject.toml name)"

        # Check for any package in src
        for child in src_dir.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                return child, f"src/{child.name}/ (first package in src)"

        # Use src/ even without __init__.py
        return src_dir, "src/ directory"

    # Check for package at project root
    if project_name:
        pkg_dir = project_root / project_name
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
            return pkg_dir, f"{project_name}/ (from pyproject.toml name)"

    # Look for any directory with __init__.py at project root
    for child in project_root.iterdir():
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and child.name not in ("tests", "test", "docs", "venv", ".venv", "env", "node_modules")
        ):
            if (child / "__init__.py").exists():
                return child, f"{child.name}/ (has __init__.py)"

    # Default to project root
    return project_root, "project root (no package structure detected)"


def _detect_js_module_root(project_root: Path) -> tuple[Path, str]:
    """Detect JavaScript/TypeScript module root.

    Priority:
    1. src/, lib/, source/ directories (common source directories)
    2. package.json "exports" field (if not in build output directory)
    3. package.json "module" field (ESM, if not in build output directory)
    4. package.json "main" field (CJS, if not in build output directory)
    5. Project root

    Build output directories (build/, dist/, out/) are skipped since they contain
    compiled code, not source files.

    """
    # Check for common source directories first - these are always preferred
    for src_dir in ["src", "lib", "source"]:
        if (project_root / src_dir).is_dir():
            return project_root / src_dir, f"{src_dir}/ directory"

    package_json_path = project_root / "package.json"
    package_data: dict[str, Any] = {}

    if package_json_path.exists():
        try:
            with package_json_path.open(encoding="utf8") as f:
                package_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Check exports field (modern Node.js)
    exports = package_data.get("exports")
    if exports:
        entry_path = _extract_entry_path(exports)
        if entry_path:
            parent = Path(entry_path).parent
            if (
                parent != Path()
                and parent.as_posix() != "."
                and (project_root / parent).is_dir()
                and not is_build_output_dir(parent)
            ):
                return project_root / parent, f'{parent.as_posix()}/ (from package.json "exports")'

    # Check module field (ESM)
    module_field = package_data.get("module")
    if module_field and isinstance(module_field, str):
        parent = Path(module_field).parent
        if (
            parent != Path()
            and parent.as_posix() != "."
            and (project_root / parent).is_dir()
            and not is_build_output_dir(parent)
        ):
            return project_root / parent, f'{parent.as_posix()}/ (from package.json "module")'

    # Check main field (CJS)
    main_field = package_data.get("main")
    if main_field and isinstance(main_field, str):
        parent = Path(main_field).parent
        if (
            parent != Path()
            and parent.as_posix() != "."
            and (project_root / parent).is_dir()
            and not is_build_output_dir(parent)
        ):
            return project_root / parent, f'{parent.as_posix()}/ (from package.json "main")'

    # Default to project root
    return project_root, "project root"


def _detect_java_module_root(project_root: Path) -> tuple[Path, str]:
    """Detect Java source root directory.

    Priority:
    1. src/main/java (standard Maven/Gradle layout)
    2. src/ directory
    3. Project root

    """
    # Standard Maven/Gradle layout
    standard_src = project_root / "src" / "main" / "java"
    if standard_src.is_dir():
        return standard_src, "src/main/java (Maven/Gradle standard)"

    # Try to detect from pom.xml
    import xml.etree.ElementTree as ET

    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            ns = {"m": "http://maven.apache.org/POM/4.0.0"}
            source_dir = root.find(".//m:sourceDirectory", ns)
            if source_dir is not None and source_dir.text:
                src_path = project_root / source_dir.text
                if src_path.is_dir():
                    return src_path, f"{source_dir.text} (from pom.xml)"
        except ET.ParseError:
            pass

    # Fallback to src directory
    if (project_root / "src").is_dir():
        return project_root / "src", "src/ directory"

    return project_root, "project root"


def is_build_output_dir(path: Path) -> bool:
    """Check if a path is within a common build output directory.

    Build output directories contain compiled code and should be skipped
    in favor of source directories.

    """
    return not _BUILD_DIRS.isdisjoint(path.parts)


def _extract_entry_path(exports: Any) -> str | None:
    """Extract entry path from package.json exports field."""
    if isinstance(exports, str):
        return exports
    if isinstance(exports, dict):
        # Handle {"." : "./src/index.js"} or {".": {"import": "./src/index.js"}}
        main_export = exports.get(".") or exports.get("import") or exports.get("default")
        if isinstance(main_export, str):
            return main_export
        if isinstance(main_export, dict):
            return main_export.get("import") or main_export.get("default") or main_export.get("require")
    return None


def _detect_tests_root(project_root: Path, language: str) -> tuple[Path | None, str]:
    """Detect the tests directory.

    Common patterns:
    - tests/ or test/
    - __tests__/ (JavaScript)
    - spec/ (Ruby/JavaScript)

    """
    # Java: standard Maven/Gradle test layout
    if language == "java":
        import xml.etree.ElementTree as ET

        standard_test = project_root / "src" / "test" / "java"
        if standard_test.is_dir():
            return standard_test, "src/test/java (Maven/Gradle standard)"

        # Check for multi-module Maven project with a test module
        # that has a custom testSourceDirectory
        for test_module_name in ["test", "tests"]:
            test_module_dir = project_root / test_module_name
            test_module_pom = test_module_dir / "pom.xml"
            if test_module_pom.exists():
                try:
                    tree = ET.parse(test_module_pom)
                    root = tree.getroot()
                    ns = {"m": "http://maven.apache.org/POM/4.0.0"}
                    for build in [root.find("m:build", ns), root.find("build")]:
                        if build is not None:
                            for elem in [build.find("m:testSourceDirectory", ns), build.find("testSourceDirectory")]:
                                if elem is not None and elem.text:
                                    # Resolve ${project.basedir}/src -> test_module_dir/src
                                    dir_text = elem.text.strip().replace("${project.basedir}/", "").replace("${project.basedir}", ".")
                                    resolved = test_module_dir / dir_text
                                    if resolved.is_dir():
                                        return resolved, f"{test_module_name}/{dir_text} (from {test_module_name}/pom.xml testSourceDirectory)"
                except ET.ParseError:
                    pass
                # Test module exists but no custom testSourceDirectory - use the module root
                if test_module_dir.is_dir():
                    return test_module_dir, f"{test_module_name}/ directory (Maven test module)"

        if (project_root / "test").is_dir():
            return project_root / "test", "test/ directory"
        if (project_root / "tests").is_dir():
            return project_root / "tests", "tests/ directory"
        return project_root / "src" / "test" / "java", "src/test/java (default)"

    # Common test directory names
    test_dirs = ["tests", "test", "__tests__", "spec"]

    for test_dir in test_dirs:
        test_path = project_root / test_dir
        if test_path.is_dir():
            return test_path, f"{test_dir}/ directory"

    # For Python, check if tests are alongside source
    if language == "python":
        # Look for test_*.py files in project root
        test_files = list(project_root.glob("test_*.py"))
        if test_files:
            return project_root, "test files in project root"

    # For JS/TS, check for *.test.js or *.spec.js files
    if language in ("javascript", "typescript"):
        test_patterns = ["*.test.js", "*.test.ts", "*.spec.js", "*.spec.ts"]
        for pattern in test_patterns:
            test_files = list(project_root.rglob(pattern))
            if test_files:
                # Find common parent
                return project_root, f"found {pattern} files"

    return None, "not detected"


def _detect_test_runner(project_root: Path, language: str) -> tuple[str, str]:
    """Detect the test runner.

    Python: pytest > unittest
    JavaScript: vitest > jest > mocha

    """
    if language in ("javascript", "typescript"):
        return _detect_js_test_runner(project_root)
    if language == "java":
        return _detect_java_test_runner(project_root)
    return _detect_python_test_runner(project_root)


def _detect_java_test_runner(project_root: Path) -> tuple[str, str]:
    """Detect Java test framework."""
    import xml.etree.ElementTree as ET

    pom_path = project_root / "pom.xml"
    if pom_path.exists():
        try:
            content = pom_path.read_text(encoding="utf-8")
            if "junit-jupiter" in content or "junit.jupiter" in content:
                return "junit5", "from pom.xml (JUnit Jupiter)"
            if "testng" in content.lower():
                return "testng", "from pom.xml (TestNG)"
            if "junit" in content.lower():
                return "junit4", "from pom.xml (JUnit)"
        except Exception:
            pass

    gradle_file = project_root / "build.gradle"
    if not gradle_file.exists():
        gradle_file = project_root / "build.gradle.kts"
    if gradle_file.exists():
        try:
            content = gradle_file.read_text(encoding="utf-8")
            if "junit-jupiter" in content or "useJUnitPlatform" in content:
                return "junit5", "from build.gradle (JUnit 5)"
            if "testng" in content.lower():
                return "testng", "from build.gradle (TestNG)"
            if "junit" in content.lower():
                return "junit4", "from build.gradle (JUnit)"
        except Exception:
            pass

    return "junit5", "default (JUnit 5)"


def _detect_python_test_runner(project_root: Path) -> tuple[str, str]:
    """Detect Python test runner."""
    # Check for pytest markers
    pytest_markers = ["pytest.ini", "pyproject.toml", "conftest.py", "setup.cfg"]
    for marker in pytest_markers:
        marker_path = project_root / marker
        if marker_path.exists():
            if marker == "pyproject.toml":
                # Check for [tool.pytest] section
                try:
                    with marker_path.open("rb") as f:
                        data = tomlkit.parse(f.read())
                    if "tool" in data and "pytest" in data["tool"]:
                        return "pytest", "pyproject.toml [tool.pytest]"
                except Exception:
                    pass
            elif marker == "conftest.py":
                return "pytest", "conftest.py found"
            elif marker in ("pytest.ini", "setup.cfg"):
                # Check for pytest section in setup.cfg
                if marker == "setup.cfg":
                    try:
                        content = marker_path.read_text(encoding="utf8")
                        if "[tool:pytest]" in content or "[pytest]" in content:
                            return "pytest", "setup.cfg [pytest]"
                    except Exception:
                        pass
                else:
                    return "pytest", "pytest.ini found"

    # Default to pytest (most common)
    return "pytest", "default"


def _detect_js_test_runner(project_root: Path) -> tuple[str, str]:
    """Detect JavaScript test runner."""
    package_json_path = project_root / "package.json"

    if not package_json_path.exists():
        return "jest", "default (no package.json)"

    try:
        with package_json_path.open(encoding="utf8") as f:
            package_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "jest", "default (invalid package.json)"

    runners = ["vitest", "jest", "mocha"]
    dev_deps = package_data.get("devDependencies", {})
    deps = package_data.get("dependencies", {})
    all_deps = {**deps, **dev_deps}

    # Check dependencies (order matters - prefer more modern runners)
    for runner in runners:
        if runner in all_deps:
            return runner, "from devDependencies"

    # Parse scripts.test for hints
    scripts = package_data.get("scripts", {})
    test_script = scripts.get("test", "")
    if isinstance(test_script, str):
        test_lower = test_script.lower()
        for runner in runners:
            if runner in test_lower:
                return runner, "from scripts.test"

    # Check for config files
    config_files = {
        "vitest": ["vitest.config.js", "vitest.config.ts", "vitest.config.mjs"],
        "jest": ["jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.json"],
        "mocha": [".mocharc.js", ".mocharc.json", ".mocharc.yaml"],
    }
    for runner, configs in config_files.items():
        for config in configs:
            if (project_root / config).exists():
                return runner, f"{config} found"

    return "jest", "default"


def _detect_formatter(project_root: Path, language: str) -> tuple[list[str], str]:
    """Detect code formatter.

    Python: ruff > black
    JavaScript: prettier > eslint --fix
    Java: google-java-format (if java and JAR available)

    """
    if language in ("javascript", "typescript"):
        return _detect_js_formatter(project_root)
    if language == "java":
        return _detect_java_formatter(project_root)
    return _detect_python_formatter(project_root)


def _detect_java_formatter(project_root: Path) -> tuple[list[str], str]:
    """Detect Java formatter (google-java-format).

    Checks for a Java executable and the google-java-format JAR in standard locations.
    Returns formatter commands if both are available, otherwise returns an empty list
    with a descriptive fallback message.

    """
    from codeflash.languages.java.formatter import JavaFormatter

    # Find java executable
    java_executable = None
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java_path = Path(java_home) / "bin" / "java"
        if java_path.exists():
            java_executable = str(java_path)
    if not java_executable:
        java_which = shutil.which("java")
        if java_which:
            java_executable = java_which

    if not java_executable:
        return [], "no Java formatter found (java not available)"

    # Check for google-java-format JAR in standard locations
    version = JavaFormatter.GOOGLE_JAVA_FORMAT_VERSION
    jar_name = f"google-java-format-{version}-all-deps.jar"
    possible_paths = [
        project_root / ".codeflash" / jar_name,
        Path.home() / ".codeflash" / jar_name,
        Path(tempfile.gettempdir()) / "codeflash" / jar_name,
    ]

    jar_path = None
    for candidate in possible_paths:
        if candidate.exists():
            jar_path = candidate
            break

    if not jar_path:
        return [], "no Java formatter found (install google-java-format)"

    return ([f"{java_executable} -jar {jar_path} --replace $file"], "google-java-format")


def _detect_python_formatter(project_root: Path) -> tuple[list[str], str]:
    """Detect Python formatter."""
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        try:
            with pyproject_path.open("rb") as f:
                data = tomlkit.parse(f.read())

            tool = data.get("tool", {})

            # Check for ruff
            if "ruff" in tool:
                return ["ruff check --exit-zero --fix $file", "ruff format $file"], "from pyproject.toml [tool.ruff]"

            # Check for black
            if "black" in tool:
                return ["black $file"], "from pyproject.toml [tool.black]"
        except Exception:
            pass

    # Check for config files
    if (project_root / "ruff.toml").exists() or (project_root / ".ruff.toml").exists():
        return ["ruff check --exit-zero --fix $file", "ruff format $file"], "ruff.toml found"

    if (project_root / ".black").exists() or (project_root / "pyproject.toml").exists():
        # Default to black if pyproject.toml exists (common setup)
        return ["black $file"], "default (black)"

    return [], "none detected"


def _detect_js_formatter(project_root: Path) -> tuple[list[str], str]:
    """Detect JavaScript formatter."""
    package_json_path = project_root / "package.json"

    # Check for prettier config files
    prettier_configs = [".prettierrc", ".prettierrc.js", ".prettierrc.json", "prettier.config.js"]
    for config in prettier_configs:
        if (project_root / config).exists():
            return ["npx prettier --write $file"], f"{config} found"

    # Check for eslint config files
    eslint_configs = [".eslintrc", ".eslintrc.js", ".eslintrc.json", "eslint.config.js"]
    for config in eslint_configs:
        if (project_root / config).exists():
            return ["npx eslint --fix $file"], f"{config} found"

    # Check package.json dependencies
    if package_json_path.exists():
        try:
            with package_json_path.open(encoding="utf8") as f:
                package_data = json.load(f)

            dev_deps = package_data.get("devDependencies", {})
            deps = package_data.get("dependencies", {})
            all_deps = {**deps, **dev_deps}

            if "prettier" in all_deps:
                return ["npx prettier --write $file"], "from devDependencies"
            if "eslint" in all_deps:
                return ["npx eslint --fix $file"], "from devDependencies"
        except (json.JSONDecodeError, OSError):
            pass

    return [], "none detected"


def _detect_ignore_paths(project_root: Path, language: str) -> tuple[list[Path], str]:
    """Detect paths to ignore during optimization.

    Sources:
    1. .gitignore
    2. Language-specific defaults

    """
    ignore_paths: list[Path] = []
    sources: list[str] = []

    # Default ignore patterns by language
    default_ignores: dict[str, list[str]] = {
        "python": [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "venv",
            ".venv",
            "env",
            ".env",
            "dist",
            "build",
            "*.egg-info",
            ".tox",
            ".nox",
            "htmlcov",
            ".coverage",
        ],
        "javascript": ["node_modules", "dist", "build", ".next", ".nuxt", "coverage", ".cache"],
        "typescript": ["node_modules", "dist", "build", ".next", ".nuxt", "coverage", ".cache"],
        "java": ["target", "build", ".gradle", ".idea", "out"],
    }

    # Add default ignores
    for pattern in default_ignores.get(language, []):
        path = project_root / pattern.replace("*", "")
        if path.exists():
            ignore_paths.append(path)

    if ignore_paths:
        sources.append("defaults")

    # Parse .gitignore
    gitignore_path = project_root / ".gitignore"
    if gitignore_path.exists():
        try:
            content = gitignore_path.read_text(encoding="utf8")
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Skip negation patterns
                if line.startswith("!"):
                    continue
                # Convert gitignore pattern to path
                pattern = line.rstrip("/").lstrip("/")
                # Skip complex patterns for now
                if "*" in pattern or "?" in pattern:
                    continue
                path = project_root / pattern
                if path.exists() and path not in ignore_paths:
                    ignore_paths.append(path)

            if ".gitignore" not in sources:
                sources.append(".gitignore")
        except Exception:
            pass

    detail = " + ".join(sources) if sources else "none"
    return ignore_paths, detail


def has_existing_config(project_root: Path) -> tuple[bool, str | None]:
    """Check if project has existing Codeflash configuration.

    Args:
        project_root: Root directory of the project.

    Returns:
        Tuple of (has_config, config_file_type).
        config_file_type is "pyproject.toml", "codeflash.toml", "package.json", or None.

    """
    # Check TOML config files (pyproject.toml, codeflash.toml)
    for toml_filename in ("pyproject.toml", "codeflash.toml"):
        toml_path = project_root / toml_filename
        if toml_path.exists():
            try:
                with toml_path.open("rb") as f:
                    data = tomlkit.parse(f.read())
                if "tool" in data and "codeflash" in data["tool"]:
                    return True, toml_filename
            except Exception:
                pass

    # Check package.json
    package_json_path = project_root / "package.json"
    if package_json_path.exists():
        try:
            with package_json_path.open(encoding="utf8") as f:
                data = json.load(f)
            if "codeflash" in data:
                return True, "package.json"
        except Exception:
            pass

    return False, None
