import os
import pathlib
import re
import subprocess


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

    print(f"Coverage was {coverage} and executed lines were {executed_lines}, as expected")

    add_one_level_depth_coverage_search = re.search(
        r"main_func_coverage=FunctionCoverage\(\n\s+name='add_one_level_depth',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )

    add_one_level_depth_coverage = float(add_one_level_depth_coverage_search.group(1))

    assert add_one_level_depth_coverage == 100.0, f"Coverage was {add_one_level_depth_coverage} instead of 100.0"

    add_one_level_depth_executed_lines = list(map(int, add_one_level_depth_coverage_search.group(2).split(", ")))

    assert add_one_level_depth_executed_lines == [
        41
    ], f"Executed lines were {add_one_level_depth_executed_lines} instead of [41]"

    print(
        f"Coverage was {add_one_level_depth_coverage} and executed lines were {add_one_level_depth_executed_lines}, as expected"
    )

    add_one_level_depth_dependent_coverage_search = re.search(
        r"dependent_func_coverage=FunctionCoverage\(\n\s+name='add',\n\s+coverage=([\d.]+),\n\s+executed_lines=\[(.+)\],",
        stdout,
    )

    add_one_level_depth_dependent_coverage = float(add_one_level_depth_dependent_coverage_search.group(1))

    assert (
        add_one_level_depth_dependent_coverage == 100.0
    ), f"Coverage was {add_one_level_depth_dependent_coverage} instead of 100.0"

    add_one_level_depth_dependent_executed_lines = list(
        map(int, add_one_level_depth_dependent_coverage_search.group(2).split(", "))
    )

    assert add_one_level_depth_dependent_executed_lines == [
        44
    ], f"Executed lines were {add_one_level_depth_dependent_executed_lines} instead of [44]"


def main() -> None:
    mybestrepocoverage()


if __name__ == "__main__":
    main()
