from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def find_node_executable() -> Optional[Path]:
    node_path = shutil.which("node")
    if node_path:
        return Path(node_path)

    npx_path = shutil.which("npx")
    if npx_path:
        return Path(npx_path)

    return None


def find_trace_runner() -> Optional[Path]:
    cwd = Path.cwd()

    local_path = cwd / "node_modules" / "codeflash" / "runtime" / "trace-runner.js"
    if local_path.exists():
        return local_path

    try:
        result = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True, check=True)
        global_modules = Path(result.stdout.strip())
        global_path = global_modules / "codeflash" / "runtime" / "trace-runner.js"
        if global_path.exists():
            return global_path
    except Exception:
        pass

    bundled_path = Path(__file__).parent.parent.parent.parent / "packages" / "codeflash" / "runtime" / "trace-runner.js"
    if bundled_path.exists():
        return bundled_path

    return None


def run_javascript_tracer(args: Namespace, config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"success": False, "trace_file": None, "replay_test_file": None, "error": None}

    node_path = find_node_executable()
    if not node_path:
        result["error"] = "Node.js not found. Please install Node.js to use JavaScript tracing."
        logger.error(result["error"])
        return result

    trace_runner_path = find_trace_runner()
    if not trace_runner_path:
        result["error"] = "trace-runner.js not found. Please install the codeflash npm package."
        logger.error(result["error"])
        return result

    outfile = getattr(args, "outfile", None) or "codeflash.trace.sqlite"
    trace_file = Path(outfile).resolve()

    env = os.environ.copy()
    env["CODEFLASH_TRACE_DB"] = str(trace_file)
    env["CODEFLASH_PROJECT_ROOT"] = str(project_root)

    max_count = getattr(args, "max_function_count", 256)
    env["CODEFLASH_MAX_FUNCTION_COUNT"] = str(max_count)

    timeout = getattr(args, "tracer_timeout", None)
    if timeout:
        env["CODEFLASH_TRACER_TIMEOUT"] = str(timeout)

    only_functions = getattr(args, "only_functions", None)
    if only_functions:
        env["CODEFLASH_FUNCTIONS"] = json.dumps(only_functions)

    cmd = [str(node_path), str(trace_runner_path)]

    cmd.extend(["--trace-db", str(trace_file)])
    cmd.extend(["--project-root", str(project_root)])

    if max_count:
        cmd.extend(["--max-function-count", str(max_count)])

    if timeout:
        cmd.extend(["--timeout", str(timeout)])

    if only_functions:
        cmd.extend(["--functions", json.dumps(only_functions)])

    is_module = getattr(args, "module", False)
    script_args = []

    if hasattr(args, "script_args"):
        script_args = args.script_args
    elif hasattr(args, "unknown_args"):
        script_args = args.unknown_args

    if is_module and script_args and script_args[0] == "jest":
        cmd.append("--jest")
        cmd.append("--")
        cmd.extend(script_args[1:])
    elif is_module and script_args and script_args[0] == "vitest":
        cmd.append("--vitest")
        cmd.append("--")
        cmd.extend(script_args[1:])
    elif script_args:
        cmd.extend(script_args)

    logger.info("Running JavaScript tracer: %s", " ".join(cmd))

    try:
        process = subprocess.run(cmd, cwd=project_root, env=env, capture_output=False, check=False)

        if process.returncode != 0:
            result["error"] = f"Tracing failed with exit code {process.returncode}"
            logger.error(result["error"])
            return result

    except Exception as e:
        result["error"] = f"Failed to run tracer: {e}"
        logger.exception(result["error"])
        return result

    if not trace_file.exists():
        result["error"] = f"Trace file not created: {trace_file}"
        logger.error(result["error"])
        return result

    result["success"] = True
    result["trace_file"] = str(trace_file)

    trace_only = getattr(args, "trace_only", False)
    if not trace_only:
        replay_test_path = generate_replay_test(trace_file=trace_file, project_root=project_root, config=config)
        if replay_test_path:
            result["replay_test_file"] = str(replay_test_path)
            logger.info("Generated replay test: %s", replay_test_path)

    return result


def generate_replay_test(
    trace_file: Path, project_root: Path, config: dict[str, Any], output_path: Optional[Path] = None
) -> Optional[Path]:
    from codeflash.languages.javascript.replay_test import create_replay_test_file

    framework = detect_test_framework(project_root, config)

    if output_path is None:
        tests_root = config.get("tests_root", "tests")
        tests_dir = project_root / tests_root
        output_path = tests_dir / "codeflash_replay.test.js"

    return create_replay_test_file(
        trace_file=trace_file,
        output_path=output_path,
        framework=framework,
        max_run_count=100,
        project_root=project_root,
    )


def detect_test_framework(project_root: Path, config: dict[str, Any]) -> str:
    if "test_framework" in config:
        framework: str = config["test_framework"]
        return framework

    vitest_configs = ["vitest.config.js", "vitest.config.ts", "vitest.config.mjs"]
    for conf in vitest_configs:
        if (project_root / conf).exists():
            return "vitest"

    jest_configs = ["jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.json"]
    for conf in jest_configs:
        if (project_root / conf).exists():
            return "jest"

    package_json = project_root / "package.json"
    if package_json.exists():
        try:
            with package_json.open(encoding="utf-8") as f:
                pkg = json.load(f)
                test_script = pkg.get("scripts", {}).get("test", "")
                if "vitest" in test_script:
                    return "vitest"
                if "jest" in test_script:
                    return "jest"

                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "vitest" in deps:
                    return "vitest"
                if "jest" in deps:
                    return "jest"
        except Exception:
            pass

    return "jest"


def check_javascript_tracer_available() -> bool:
    if not find_node_executable():
        return False

    if not find_trace_runner():
        return False

    return True


def get_tracer_requirements_message() -> str:
    missing = []

    if not find_node_executable():
        missing.append("Node.js (v18+)")

    if not find_trace_runner():
        missing.append("codeflash npm package (npm install codeflash)")

    if not missing:
        return "All requirements met for JavaScript tracing."

    return "Missing requirements for JavaScript tracing:\n- " + "\n- ".join(missing)
