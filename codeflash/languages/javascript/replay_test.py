from __future__ import annotations

import json
import re
import sqlite3
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class JavaScriptFunctionModule:
    function_name: str
    file_name: Path
    module_name: str
    class_name: Optional[str] = None
    line_no: Optional[int] = None


def get_next_arg_and_return(
    trace_file: str, function_name: str, file_name: str, class_name: Optional[str] = None, num_to_get: int = 25
) -> Generator[Any]:
    db = sqlite3.connect(trace_file)
    cur = db.cursor()

    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        if "function_calls" in tables:
            if class_name:
                cursor = cur.execute(
                    "SELECT args FROM function_calls WHERE function = ? AND filename = ? AND classname = ? AND type = 'call' ORDER BY time_ns ASC LIMIT ?",
                    (function_name, file_name, class_name, num_to_get),
                )
            else:
                cursor = cur.execute(
                    "SELECT args FROM function_calls WHERE function = ? AND filename = ? AND type = 'call' ORDER BY time_ns ASC LIMIT ?",
                    (function_name, file_name, num_to_get),
                )

            while (val := cursor.fetchone()) is not None:
                args_data = val[0]
                if isinstance(args_data, bytes):
                    yield args_data
                else:
                    yield args_data

        elif "traces" in tables:
            if class_name:
                cursor = cur.execute(
                    "SELECT args FROM traces WHERE function = ? AND file = ? ORDER BY id ASC LIMIT ?",
                    (function_name, file_name, num_to_get),
                )
            else:
                cursor = cur.execute(
                    "SELECT args FROM traces WHERE function = ? AND file = ? ORDER BY id ASC LIMIT ?",
                    (function_name, file_name, num_to_get),
                )

            while (val := cursor.fetchone()) is not None:
                yield val[0]

    finally:
        db.close()


def get_function_alias(module: str, function_name: str, class_name: Optional[str] = None) -> str:
    module_alias = re.sub(r"[^a-zA-Z0-9]", "_", module).strip("_")

    if class_name:
        return f"{module_alias}_{class_name}_{function_name}"
    return f"{module_alias}_{function_name}"


def create_javascript_replay_test(
    trace_file: str,
    functions: list[JavaScriptFunctionModule],
    max_run_count: int = 100,
    framework: str = "jest",
    project_root: Optional[Path] = None,
) -> str:
    is_vitest = framework.lower() == "vitest"

    imports = []

    if is_vitest:
        imports.append("import { describe, test } from 'vitest';")

    imports.append("const { getNextArg } = require('codeflash/replay');")
    imports.append("")

    for func in functions:
        if func.function_name in ("__init__", "constructor"):
            continue

        alias = get_function_alias(func.module_name, func.function_name, func.class_name)

        if func.class_name:
            imports.append(f"const {{ {func.class_name}: {alias}_class }} = require('./{func.module_name}');")
        else:
            imports.append(f"const {{ {func.function_name}: {alias} }} = require('./{func.module_name}');")

    imports.append("")

    functions_to_test = [f.function_name for f in functions if f.function_name not in ("__init__", "constructor")]
    metadata = f"""const traceFilePath = '{trace_file}';
const functions = {json.dumps(functions_to_test)};
"""

    test_cases = []

    for func in functions:
        if func.function_name in ("__init__", "constructor"):
            continue

        alias = get_function_alias(func.module_name, func.function_name, func.class_name)
        test_name = f"{func.class_name}.{func.function_name}" if func.class_name else func.function_name

        if func.class_name:
            class_arg = f"'{func.class_name}'"
            test_body = textwrap.dedent(f"""
describe('Replay: {test_name}', () => {{
    const traces = getNextArg(traceFilePath, '{func.function_name}', '{func.file_name.as_posix()}', {max_run_count}, {class_arg});

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {{
        const instance = new {alias}_class();
        instance.{func.function_name}(...args);
    }});
}});
""")
        else:
            test_body = textwrap.dedent(f"""
describe('Replay: {test_name}', () => {{
    const traces = getNextArg(traceFilePath, '{func.function_name}', '{func.file_name.as_posix()}', {max_run_count});

    test.each(traces.map((args, i) => [i, args]))('call %i', (index, args) => {{
        {alias}(...args);
    }});
}});
""")

        test_cases.append(test_body)

    return "\n".join(
        [
            "// Auto-generated replay test by Codeflash",
            "// Do not edit this file directly",
            "",
            *imports,
            metadata,
            *test_cases,
        ]
    )


def get_traced_functions_from_db(trace_file: Path) -> list[JavaScriptFunctionModule]:
    if not trace_file.exists():
        return []

    try:
        conn = sqlite3.connect(trace_file)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        functions = []

        if "function_calls" in tables:
            cursor.execute(
                "SELECT DISTINCT function, filename, classname, line_number FROM function_calls WHERE type = 'call'"
            )
            for row in cursor.fetchall():
                func_name = row[0]
                file_name = row[1]
                class_name = row[2]
                line_number = row[3]

                module_path = file_name.replace("\\", "/").replace(".js", "").replace(".ts", "")
                module_path = module_path.removeprefix("./")

                functions.append(
                    JavaScriptFunctionModule(
                        function_name=func_name,
                        file_name=Path(file_name),
                        module_name=module_path,
                        class_name=class_name,
                        line_no=line_number,
                    )
                )

        elif "traces" in tables:
            cursor.execute("SELECT DISTINCT function, file FROM traces")
            for row in cursor.fetchall():
                func_name = row[0]
                file_name = row[1]

                module_path = file_name.replace("\\", "/").replace(".js", "").replace(".ts", "")
                module_path = module_path.removeprefix("./")

                functions.append(
                    JavaScriptFunctionModule(
                        function_name=func_name, file_name=Path(file_name), module_name=module_path
                    )
                )

        conn.close()
        return functions

    except Exception:
        return []


def create_replay_test_file(
    trace_file: Path,
    output_path: Path,
    framework: str = "jest",
    max_run_count: int = 100,
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    functions = get_traced_functions_from_db(trace_file)

    if not functions:
        return None

    content = create_javascript_replay_test(
        trace_file=str(trace_file),
        functions=functions,
        max_run_count=max_run_count,
        framework=framework,
        project_root=project_root,
    )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return output_path
    except Exception:
        return None
