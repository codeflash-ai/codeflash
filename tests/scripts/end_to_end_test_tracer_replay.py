import os
import pathlib
import re
import subprocess


def main():
    cwd = (
        pathlib.Path(__file__).parent.parent.parent / "code_to_optimize" / "code_directories" / "simple_tracer_e2e"
    ).resolve()
    print("cwd", cwd)
    command = ["python", "-m", "codeflash.tracer", "-o", "codeflash.trace", "workload.py"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd), env=os.environ.copy()
    )
    output = []

    for line in process.stdout:
        print(line, end="")  # Print each line in real-time
        output.append(line)  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"
    functions_traced = re.search(r"Traced (\d+) function calls successfully and replay test created at - (.*)$", stdout)
    assert functions_traced, "Failed to find any traced functions or replay test"
    assert int(functions_traced.group(1)) == 1, "Failed to find the correct number of traced functions"
    replay_test_path = pathlib.Path(functions_traced.group(2))
    assert replay_test_path, "Failed to find the replay test file path"
    assert replay_test_path.exists(), f"Replay test file does not exist at - {replay_test_path}"

    command = ["python", "../../../codeflash/main.py", "--replay-test", str(replay_test_path), "--no-pr"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(cwd), env=os.environ.copy()
    )
    output = []

    for line in process.stdout:
        print(line, end="")  # Print each line in real-time
        output.append(line)  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"

    improvement_pct = int(re.search(r"📈 ([\d,]+)% improvement", stdout).group(1).replace(",", ""))
    improvement_x = float(improvement_pct) / 100

    assert improvement_pct > 10, f"Performance improvement percentage was {improvement_pct}, which was not above 10%"
    assert improvement_x > 0.1, f"Performance improvement rate was {improvement_x}x, which was not above 0.1x"

    # Check for the line indicating the number of discovered existing unit tests
    unit_test_search = re.search(r"Discovered (\d+) existing unit tests", stdout)
    num_unit_tests = int(unit_test_search.group(1))
    assert num_unit_tests == 1, f"Could not find 1 existing unit test, found {num_unit_tests} instead"

    # check if the replay test was correctly run for the original code
    m = re.search(r"Replay Tests - Passed: (\d+), Failed: (\d+)", stdout)
    assert m, "Failed to run replay tests"

    passed, failed = int(m.group(1)), int(m.group(2))

    assert passed > 0, f"Expected >0 passed replay tests, found {passed}"
    assert failed == 0, f"Expected 0 failed replay tests, found {failed}"


if __name__ == "__main__":
    main()
