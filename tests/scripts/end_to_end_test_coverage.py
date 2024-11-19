import os
import pathlib
import re
import subprocess


def futurehouse_coverage() -> None:
    cwd = (
        pathlib.Path(__file__).parent.parent.parent / "code_to_optimize" / "code_directories" / "futurehouse_structure"
    ).resolve()

    command = ["python", "../../../codeflash/main.py", "--file", "src/aviary/common_tags.py", "--no-pr"]
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

    assert "CoverageData(" in stdout, "Failed to find CoverageData in stdout"

    coverage_search = re.search(
        r"main_func_coverage=FunctionCoverage\(\n\s+name='find_common_tags',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )
    coverage = float(coverage_search.group(1))
    assert coverage == 100.0, f"Coverage was {coverage} instead of 100.0"

    executed_lines = list(map(int, coverage_search.group(2).split(", ")))

    assert executed_lines == [
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
    ], f"Executed lines were {executed_lines} instead of [5, 6, 7, 8, 9, 11, 12, 13, 14]"

    print(f"Coverage was {coverage} and executed lines were {executed_lines}, as expected")


def mybestrepocoverage() -> None:
    cwd = (
        pathlib.Path(__file__).parent.parent.parent / "code_to_optimize" / "code_directories" / "my-best-repo"
    ).resolve()
    command = ["python", "../../../codeflash/main.py", "--file", "bubble_sort.py", "--no-pr"]
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

    assert "CoverageData(" in stdout, "Failed to find CoverageData in stdout"

    coverage_search = re.search(
        r"main_func_coverage=FunctionCoverage\(\n\s+name='sorter_one_level_depth',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )

    coverage = float(coverage_search.group(1))
    assert coverage == 100.0, f"Coverage was {coverage} instead of 100.0"

    executed_lines = list(map(int, coverage_search.group(2).split(", ")))

    assert executed_lines == [2], f"Executed lines were {executed_lines} instead of [2]"

    add_one_level_depth_coverage_search = re.search(
        r"dependent_func_coverage=FunctionCoverage\(\n\s+name='add',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )

    coverage = float(add_one_level_depth_coverage_search.group(1))
    assert coverage == 100.0, f"Coverage was {coverage} instead of 100.0"

    executed_lines = list(map(int, add_one_level_depth_coverage_search.group(2).split(", ")))

    assert executed_lines == [48], f"Executed lines were {executed_lines} instead of [48]"

    print(f"Coverage was {coverage} and executed lines were {executed_lines}, as expected")

    dependent_func_coverage_search = re.search(
        r"dependent_func_coverage=FunctionCoverage\(\n\s+name='add',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )

    coverage = float(dependent_func_coverage_search.group(1))

    assert coverage == 100.0, f"Coverage was {coverage} instead of 100.0"

    executed_lines = list(map(int, dependent_func_coverage_search.group(2).split(", ")))

    assert executed_lines == [48], f"Executed lines were {executed_lines} instead of [48]"


def main() -> None:
    futurehouse_coverage()
    mybestrepocoverage()


if __name__ == "__main__":
    main()
