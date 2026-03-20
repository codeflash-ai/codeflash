import logging
import os
import pathlib
import re
import shutil
import subprocess
import time


def run_test(expected_improvement_pct: int) -> bool:
    logging.basicConfig(level=logging.INFO)
    fixture_dir = (pathlib.Path(__file__).parent.parent / "test_languages" / "fixtures" / "java_tracer_e2e").resolve()

    # Ensure test directory exists (git doesn't track empty dirs)
    test_java_dir = fixture_dir / "src" / "test" / "java"
    test_java_dir.mkdir(parents=True, exist_ok=True)

    # Clean up leftover replay tests from previous runs
    replay_dir = test_java_dir / "codeflash" / "replay"
    if replay_dir.exists():
        shutil.rmtree(replay_dir, ignore_errors=True)
    for f in test_java_dir.rglob("*__perfinstrumented*.java"):
        f.unlink(missing_ok=True)
    for f in test_java_dir.rglob("*__perfonlyinstrumented*.java"):
        f.unlink(missing_ok=True)

    # Compile the workload
    classes_dir = fixture_dir / "target" / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    compile_result = subprocess.run(
        [
            "javac",
            "--release",
            "11",
            "-d",
            str(classes_dir),
            str(fixture_dir / "src" / "main" / "java" / "com" / "example" / "Workload.java"),
        ],
        capture_output=True,
        text=True,
    )
    if compile_result.returncode != 0:
        logging.error(f"javac failed: {compile_result.stderr}")
        return False

    # Run the Java tracer + optimizer
    command = [
        "uv",
        "run",
        "--no-project",
        "-m",
        "codeflash.main",
        "optimize",
        "java",
        "-cp",
        str(classes_dir),
        "com.example.Workload",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    logging.info(f"Running command: {' '.join(command)}")
    logging.info(f"Working directory: {fixture_dir}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(fixture_dir),
        env=env,
        encoding="utf-8",
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)
    if return_code != 0:
        logging.error(f"Full output:\n{stdout}")

    if return_code != 0:
        logging.error(f"Command returned exit code {return_code}")
        return False

    # Validate: replay tests were generated
    if "replay test files generated" not in stdout:
        logging.error("Failed to find replay test generation message")
        return False

    # Validate: replay tests were discovered (global count)
    replay_match = re.search(r"Discovered \d+ existing unit tests? and (\d+) replay tests?", stdout)
    if not replay_match:
        logging.error("Failed to find replay test discovery message")
        return False
    num_replay = int(replay_match.group(1))
    if num_replay == 0:
        logging.error("No replay tests discovered")
        return False
    logging.info(f"Replay tests discovered: {num_replay}")

    # Validate: replay test files were used per-function
    replay_file_match = re.search(r"Discovered \d+ existing unit test files?, (\d+) replay test files?", stdout)
    if not replay_file_match:
        logging.error("Failed to find per-function replay test file discovery message")
        return False
    num_replay_files = int(replay_file_match.group(1))
    if num_replay_files == 0:
        logging.error("No replay test files discovered per-function")
        return False
    logging.info(f"Replay test files per-function: {num_replay_files}")

    # Validate: at least one optimization was found
    if "⚡️ Optimization successful! 📄 " not in stdout:
        logging.error("Failed to find optimization success message")
        return False

    improvement_match = re.search(r"📈 ([\d,]+)% (?:(\w+) )?improvement", stdout)
    if not improvement_match:
        logging.error("Could not find improvement percentage in output")
        return False

    improvement_pct = int(improvement_match.group(1).replace(",", ""))
    logging.info(f"Performance improvement: {improvement_pct}%")

    if improvement_pct <= expected_improvement_pct:
        logging.error(f"Performance improvement {improvement_pct}% not above {expected_improvement_pct}%")
        return False

    logging.info(f"Success: Java tracer e2e passed with {improvement_pct}% improvement")
    return True


def run_with_retries(test_func, *args) -> int:
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    retry_delay = int(os.getenv("RETRY_DELAY", 5))
    for attempt in range(1, max_retries + 1):
        logging.info(f"\n=== Attempt {attempt} of {max_retries} ===")
        if test_func(*args):
            logging.info(f"Test passed on attempt {attempt}")
            return 0
        logging.error(f"Test failed on attempt {attempt}")
        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logging.error("Test failed after all retries")
            return 1
    return 1


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 10))))
