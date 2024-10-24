import os
import pathlib
import re
import subprocess


def main():
    cwd = (
        pathlib.Path(__file__).parent.parent.parent
        / "code_to_optimize"
        / "code_directories"
        / "futurehouse_structure"
    ).resolve()
    print("cwd", cwd)
    command = [
        "python",
        "../codeflash/main.py",
        "--all",
        "--no-pr",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd),
        env=os.environ.copy(),
    )
    output = []

    for line in process.stdout:
        print(line, end="")  # Print each line in real-time
        output.append(line)  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"

    assert "⚡️ Optimization successful! 📄 " in stdout, "Failed to find performance improvement at all"

    improvement_pct = int(re.search(r"📈 ([\d,]+)% improvement", stdout).group(1))
    improvement_x = float(improvement_pct) / 100

    assert (
        improvement_pct > 5
    ), f"Performance improvement percentage was {improvement_pct}, which was not above 10%"
    assert improvement_x > 0.1, f"Performance improvement rate was {improvement_x}x, which was not above 0.1x"

    # Check for the line indicating the number of discovered existing unit tests
    unit_test_search = re.search(
        r"Discovered (\d+) existing unit tests",
        stdout,
    )
    num_unit_tests = int(unit_test_search.group(1))
    assert num_unit_tests == 2, "Could not find existing unit tests"


if __name__ == "__main__":
    main()
