import io
import json
import subprocess
import unittest.mock
from pathlib import Path


def create_files() -> None:
    problems = set()
    for jsonl_file in [
        "original_data/test.jsonl",
        "original_data/val.jsonl",
        "original_data/train.jsonl",
    ]:
        jsonl_path = Path(jsonl_file)
        if not jsonl_path.exists():
            print(f"File {jsonl_file} does not exist.")
            continue
        with jsonl_path.open("r") as file:
            for line in file:
                test_case = json.loads(line)
                problem_id = test_case["problem_id"]
                input_code = test_case["input"]
                if problem_id in problems:
                    continue

                problems.add(problem_id)

                # Create a new Python file for each problem_id
                file_path = Path(f"../{problem_id}.py")
                if file_path.exists():
                    print(f"File {file_path} already exists.")
                    continue

                with file_path.open("w") as code_file:
                    # Write the input code into a new function in the file
                    code_file.write(f"def problem_{problem_id}(input_data):\n")
                    # Replace input() calls with input_data handling
                    indented_code = "    " + input_code.replace("input()", "input_data").replace(
                        "read()", "input_data"
                    ).replace("print", "return ").replace("\n", "\n    ")
                    code_file.write(indented_code + "\n")
                    # Return the result instead of printing it
                    # Ensure the result is returned instead of printed
                try:
                    # Run black to reformat the code file
                    subprocess.run(["black", str(file_path)], check=True)
                except subprocess.CalledProcessError:
                    print(f"Failed to format {file_path} with black.")
                    file_path.unlink()
                    continue

                # Create test cases for each problem_id
                test_dir = Path(f"../tests/public_test_cases/{problem_id}")
                if not test_dir.exists():
                    print(f"Directory {test_dir} does not exist.")
                    continue
                test_files = sorted(
                    [f for f in test_dir.iterdir() if f.name.startswith("input")],
                    key=lambda x: int(x.stem.split(".")[1]),
                )
                test_code_file_path = Path(f"../tests/test_{problem_id}.py")
                with test_code_file_path.open("w") as test_code_file:
                    test_code_file.write("\n")
                    test_code_file.write(
                        f"from code_to_optimize.pie_test_set.{problem_id} import problem_{problem_id}\n\n"
                    )
                for test_file in test_files:
                    input_num = test_file.stem.split(".")[1]
                    output_file = test_dir / f"output.{input_num}.txt"
                    with test_file.open("r") as input_f, output_file.open("r") as output_f:
                        input_content = input_f.read()
                        expected_output = output_f.read()
                        if "\n" in input_content.strip() or "\n" in expected_output.strip():
                            print(f"Multiple lines detected in input or output for {problem_id}, skipping.")
                            file_path.unlink()
                            if test_code_file_path.exists():
                                test_code_file_path.unlink()
                            break
                        else:
                            input_content = input_content.strip()
                            expected_output = expected_output.strip()
                            test_case_code = generate_test_case_code(
                                problem_id, input_num, input_content, expected_output
                            )
                            with test_code_file_path.open("a") as test_code_file:
                                test_code_file.write(test_case_code)
                        try:
                            # Run black to reformat the test file
                            subprocess.run(["black", str(test_code_file_path)], check=True)
                        except subprocess.CalledProcessError:
                            print(f"Failed to format {test_code_file_path} with black.")
                            test_code_file_path.unlink()
                            break


def generate_test_case_code(problem_id: str, input_num: str, input_content: str, expected_output: str) -> str:
    return (
        f"def test_problem_{problem_id}_{input_num}():\n"
        f"    actual_output = problem_{problem_id}({input_content!r})\n"
        + f"    expected_output = {expected_output!r}\n"
        + f"    assert str(actual_output) == expected_output\n\n"
    )


if __name__ == "__main__":
    create_files()
