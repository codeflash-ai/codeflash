import json
import os
import subprocess
import unittest.mock
import io


def create_files() -> None:
    problems = set()
    for jsonl_file in [
        "original_data/test.jsonl",
        "original_data/val.jsonl",
        "original_data/train.jsonl",
    ]:
        if not os.path.exists(jsonl_file):
            print(f"File {jsonl_file} does not exist.")
            continue
        with open(jsonl_file, "r") as file:
            for line in file:
                test_case = json.loads(line)
                problem_id = test_case["problem_id"]
                input_code = test_case["input"]
                if problem_id in problems:
                    continue

                problems.add(problem_id)

                # Create a new Python file for each problem_id
                file_path = f"../{problem_id}.py"
                if os.path.exists(file_path):
                    print(f"File {file_path} already exists.")
                    continue

                with open(file_path, "w") as code_file:
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
                    subprocess.run(["black", file_path], check=True)
                except subprocess.CalledProcessError:
                    print(f"Failed to format {file_path} with black.")
                    os.remove(file_path)
                    continue

                # Create test cases for each problem_id
                test_dir = f"../tests/public_test_cases/{problem_id}"
                if not os.path.exists(test_dir):
                    print(f"Directory {test_dir} does not exist.")
                    continue
                test_files = sorted(
                    [f for f in os.listdir(test_dir) if f.startswith("input")],
                    key=lambda x: int(x.split(".")[1]),
                )
                test_code_file_path = f"../tests/test_{problem_id}.py"
                with open(test_code_file_path, "w") as test_code_file:
                    test_code_file.write("\n")
                    test_code_file.write(
                        f"from code_to_optimize.pie_test_set.{problem_id} import problem_{problem_id}\n\n"
                    )
                for test_file in test_files:
                    input_num = test_file.split(".")[1]
                    output_file = f"output.{input_num}.txt"
                    with open(f"{test_dir}/{test_file}", "r") as input_f, open(
                        f"{test_dir}/{output_file}", "r"
                    ) as output_f:
                        input_content = input_f.read()
                        expected_output = output_f.read()
                        if '\n' in input_content.strip() or '\n' in expected_output.strip():
                            print(f"Multiple lines detected in input or output for {problem_id}, skipping.")
                            os.remove(file_path)
                            if os.path.exists(test_code_file_path):
                                os.remove(test_code_file_path)
                            break
                        else:
                            input_content = input_content.strip()
                            expected_output = expected_output.strip()
                            test_case_code = generate_test_case_code(
                                problem_id, input_num, input_content, expected_output
                            )
                            with open(test_code_file_path, "a") as test_code_file:
                                test_code_file.write(test_case_code)
                        try:
                            # Run black to reformat the test file
                            subprocess.run(["black", test_code_file_path], check=True)
                        except subprocess.CalledProcessError:
                            print(f"Failed to format {test_code_file_path} with black.")
                            os.remove(test_code_file_path)
                            break


def generate_test_case_code(
    problem_id: str, input_num: str, input_content: str, expected_output: str
) -> str:
    return (
        f"def test_problem_{problem_id}_{input_num}():\n"
        f"    actual_output = problem_{problem_id}({input_content!r})\n" +
        f"    expected_output = {expected_output!r}\n" +
        f"    if isinstance(actual_output, type(expected_output)):\n" +
        f"        assert actual_output == expected_output\n" +
        f"    else:\n" +
        f"        # Cast expected output to the type of actual output if they differ\n" +
        f"        cast_expected_output = type(actual_output)(expected_output)\n" +
        f"        assert actual_output == cast_expected_output\n\n"
    )


if __name__ == "__main__":
    create_files()
