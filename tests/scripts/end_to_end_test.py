import os
import pathlib
import re
import subprocess


def main():
    root = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    test_root = root / "tests" / "pytest"
    print("cwd", root)
    command = [
        "python",
        "../codeflash/main.py",
        "--file",
        "bubble_sort.py",
        "--function",
        "sorter",
        "--test-root",
        str(test_root),
        "--root",
        str(root),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(root),
        env=os.environ.copy(),
    )
    output = []

    for line in process.stdout:
        print(line, end="")  # Print each line in real-time
        output.append(line)  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"

    m = re.search(r"Performance went up by (\d+\.\d+)x", stdout)
    assert m, "Failed to find performance improvement at all"
    improvement = float(m.group(1))
    assert (
        30000 < improvement < 120000
    ), f"Performance improvement was not in the expected range, got {improvement}"


if __name__ == "__main__":
    main()
