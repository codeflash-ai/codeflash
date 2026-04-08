from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def generate_replay_tests(
    trace_db_path: Path, output_dir: Path, project_root: Path, max_run_count: int = 256, test_framework: str = "junit5"
) -> int:
    """Generate JUnit replay test files from a trace SQLite database.

    Supports both JUnit 5 (default) and JUnit 4.
    Returns the number of test files generated.
    """
    if not trace_db_path.exists():
        logger.error("Trace database not found: %s", trace_db_path)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(trace_db_path))
    try:
        cursor = conn.execute(
            "SELECT DISTINCT classname, function, descriptor FROM function_calls ORDER BY classname, function"
        )
        methods = cursor.fetchall()

        # Group by class
        class_methods: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for classname, function, descriptor in methods:
            class_methods[classname].append((function, descriptor))

        test_count = 0
        all_function_names: list[str] = []

        for classname, method_list in class_methods.items():
            safe_class_name = _sanitize_identifier(classname.replace(".", "_"))
            test_class_name = f"ReplayTest_{safe_class_name}"

            test_methods_code: list[str] = []
            class_function_names: list[str] = []
            # Global test counter to avoid duplicate method names for overloaded Java methods
            method_name_counters: dict[str, int] = {}

            for method_name, descriptor in method_list:
                count_result = conn.execute(
                    "SELECT COUNT(*) FROM function_calls WHERE classname = ? AND function = ? AND descriptor = ?",
                    (classname, method_name, descriptor),
                ).fetchone()
                invocation_count = min(count_result[0], max_run_count)

                simple_class = classname.rsplit(".", 1)[-1]
                class_function_names.append(f"{simple_class}.{method_name}")
                safe_method = _sanitize_identifier(method_name)

                for i in range(invocation_count):
                    # Use a global counter per method name to avoid collisions on overloaded methods
                    test_idx = method_name_counters.get(safe_method, 0)
                    method_name_counters[safe_method] = test_idx + 1

                    escaped_descriptor = descriptor.replace('"', '\\"')
                    access = "public " if test_framework == "junit4" else ""
                    test_methods_code.append(
                        f"    @Test {access}void replay_{safe_method}_{test_idx}() throws Exception {{\n"
                        f'        helper.replay("{classname}", "{method_name}", '
                        f'"{escaped_descriptor}", {i});\n'
                        f"    }}"
                    )

            all_function_names.extend(class_function_names)

            # Generate the test file
            functions_comment = ",".join(class_function_names)
            if test_framework == "junit4":
                test_imports = "import org.junit.Test;\nimport org.junit.AfterClass;\n"
                cleanup_annotation = "@AfterClass"
                class_modifier = "public "
            else:
                test_imports = "import org.junit.jupiter.api.Test;\nimport org.junit.jupiter.api.AfterAll;\n"
                cleanup_annotation = "@AfterAll"
                class_modifier = ""

            test_content = (
                f"// codeflash:functions={functions_comment}\n"
                f"// codeflash:trace_file={trace_db_path.as_posix()}\n"
                f"// codeflash:classname={classname}\n"
                f"package codeflash.replay;\n\n"
                f"{test_imports}"
                f"import com.codeflash.ReplayHelper;\n\n"
                f"{class_modifier}class {test_class_name} {{\n"
                f"    private static final ReplayHelper helper =\n"
                f'        new ReplayHelper("{trace_db_path.as_posix()}");\n\n'
                f"    {cleanup_annotation} public static void cleanup() {{ helper.close(); }}\n\n"
                + "\n\n".join(test_methods_code)
                + "\n"
                "}\n"
            )

            test_file = output_dir / f"{test_class_name}.java"
            test_file.write_text(test_content, encoding="utf-8")
            test_count += 1
            logger.info("Generated replay test: %s (%d test methods)", test_file.name, len(test_methods_code))

    finally:
        conn.close()

    return test_count


def _sanitize_identifier(name: str) -> str:
    """Sanitize a string for use as a Java identifier."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def parse_replay_test_metadata(test_file: Path) -> dict[str, str]:
    """Parse codeflash metadata comments from a Java replay test file.

    Returns a dict with keys: functions, trace_file, classname.
    """
    metadata: dict[str, str] = {}
    try:
        with test_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("// codeflash:"):
                    if line and not line.startswith("//"):
                        break
                    continue
                key_value = line[len("// codeflash:") :]
                if "=" in key_value:
                    key, value = key_value.split("=", 1)
                    metadata[key] = value
    except Exception:
        logger.exception("Failed to parse replay test metadata from %s", test_file)
    return metadata
