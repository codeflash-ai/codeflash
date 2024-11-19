import os
import pathlib
import re
import subprocess

import pytest


def main():
    module_root = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    test_root = module_root / "tests" / "pytest"
    print("cwd", module_root)
    command = [
        "python",
        "../codeflash/main.py",
        "--file",
        "bubble_sort.py",
        "--function",
        "sorter",
        "--test-framework",
        "pytest",
        "--tests-root",
        str(test_root),
        "--module-root",
        str(module_root),
        "--no-pr",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(module_root),
        env=os.environ.copy(),
    )
    output = []

    for line in process.stdout:
        print(line, end="")  # Print each line in real-time
        output.append(line.strip())  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"

    assert "⚡️ Optimization successful! 📄 " in stdout, "Failed to find performance improvement at all"

    improvement_pct = int(re.search(r"📈 ([\d,]+)% improvement", stdout).group(1).replace(",", ""))
    improvement_x = float(improvement_pct) / 100

    assert improvement_pct > 100, f"Performance improvement percentage was {improvement_pct}, which was not above 100%"
    assert improvement_x > 1, f"Performance improvement rate was {improvement_x}x, which was not above 1x"

    # Check for the line indicating the number of discovered existing unit tests
    unit_test_search = re.search(r"Discovered (\d+) existing unit tests", stdout)
    num_unit_tests = int(unit_test_search.group(1))
    assert num_unit_tests > 0, "Could not find existing unit tests"

    pattern = r"""
    main_func_coverage=FunctionCoverage\(
        .*?coverage=(?P<coverage>[\d.]+),
        \s*executed_lines=\[(?P<executed_lines>[\d,\s]*)\]
    """

    match = re.search(pattern, stdout, re.VERBOSE)

    if match:
        coverage = float(match.group("coverage"))

        executed_lines = list(map(int, match.group("executed_lines").split(", ")))

        assert coverage == 100.0, f"Coverage was {coverage} instead of 100.0"

        assert executed_lines == [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ], f"Executed lines were {executed_lines} instead of [2, 3, 4, 6, 9]"
    else:
        pytest.fail("Failed to find coverage data in stdout")


if __name__ == "__main__":
    main()
