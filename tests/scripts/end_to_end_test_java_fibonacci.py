import logging
import os
import pathlib
import subprocess
import time


def run_test(expected_improvement_pct: int) -> bool:
    logging.basicConfig(level=logging.INFO)
    cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize" / "java").resolve()
    file_path = "src/main/java/com/example/Fibonacci.java"
    function_name = "fibonacci"

    # Save original file contents for rollback on failure
    original_contents = (cwd / file_path).read_text("utf-8")

    command = [
        "uv", "run", "--no-project", "../../codeflash/main.py",
        "--file", file_path,
        "--function", function_name,
        "--no-pr",
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logging.info(f"Running: {' '.join(command)} in {cwd}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(cwd), env=env, encoding="utf-8",
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)

    if return_code != 0:
        logging.error(f"Command returned exit code {return_code}")
        (cwd / file_path).write_text(original_contents, "utf-8")
        return False

    if "⚡️ Optimization successful! 📄 " not in stdout:
        logging.error("Failed to find optimization success message in output")
        (cwd / file_path).write_text(original_contents, "utf-8")
        return False

    logging.info("Java Fibonacci optimization succeeded")
    # Restore original file so the test is idempotent
    (cwd / file_path).write_text(original_contents, "utf-8")
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
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 70))))
