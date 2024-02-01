import os
import pathlib
import re
import subprocess


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
        "--tests-root",
        str(test_root),
        "--module-root",
        str(module_root),
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
        output.append(line)  # Store each line in the output variable
    return_code = process.wait()
    stdout = "".join(output)
    assert return_code == 0, f"The codeflash command returned exit code {return_code} instead of 0"

    m = re.search(
        r"Optimization successful! 📄 sorter in .+\n.+📈\s+([\d+,]+)% improvement \(([\d+,.]+)x faster\)\.",
        stdout,
    )
    assert m, "Failed to find performance improvement at all"
    improvement_pct = int(m.group(1).replace(",", ""))
    improvement_x = float(m.group(2).replace(",", ""))

    assert (
        12000000 > improvement_pct > 30000
    ), f"Performance improvement percentage was {improvement_pct}, which was not in the range of 30,000% to 12,000,000%"
    assert (
        12000000 > improvement_x > 30000
    ), f"Performance improvement rate was {improvement_x}x, which was not in the range of 30,000x to 12,000,000"


if __name__ == "__main__":
    main()
