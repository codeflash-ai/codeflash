import json
import os
import subprocess
from typing import Tuple


from code_to_optimize.test_set.run_pie_test_case import run_pie_test_case


def create_files() -> None:
    for jsonl_file in ["test.jsonl", "val.jsonl", "train.jsonl"]:
        if not os.path.exists(jsonl_file):
            continue
        with open(jsonl_file, "r") as file:
            for line in file:
                test_case = json.loads(line)
                problem_id = test_case["problem_id"]
                input_code = test_case["input"]

                # Create a new Python file for each problem_id
                file_path = f"{problem_id}.py"
                if os.path.exists(file_path):
                    continue

                with open(file_path, "w") as code_file:
                    # Write the input code into a new function in the file
                    code_file.write(f"def problem_{problem_id}():\n")
                    # Indent the input code and write it to the function
                    indented_code = "    " + input_code.replace("\n", "\n    ")
                    code_file.write(indented_code + "\n")
                    # Call the function at the end of the file
                    code_file.write(f"\nproblem_{problem_id}()")
                try:
                    # Run black to reformat the code file
                    subprocess.run(["black", file_path], check=True)
                except subprocess.CalledProcessError:
                    print(f"Failed to format {file_path} with black.")
                    os.remove(file_path)
                    continue

                # Create test cases for each problem_id
                test_dir = f"tests/public_test_cases/{problem_id}"
                if not os.path.exists(test_dir):
                    continue
                test_files = sorted(
                    [f for f in os.listdir(test_dir) if f.startswith("input")],
                    key=lambda x: int(x.split(".")[1]),
                )
                test_code_file_path = f"tests/test_{problem_id}.py"
                with open(test_code_file_path, "w") as test_code_file:
                    test_code_file.write(
                        "from code_to_optimize.pie-test-set.run_pie_test_case import run_pie_test_case\n\n"
                    )
                for test_file in test_files:
                    input_num = test_file.split(".")[1]
                    output_file = f"output.{input_num}.txt"
                    with open(f"{test_dir}/{test_file}", "r") as input_f, open(
                        f"{test_dir}/{output_file}", "r"
                    ) as output_f:
                        input_content = input_f.read().strip()
                        expected_output = output_f.read().strip()
                        test_case_code = generate_test_case_code(
                            problem_id, input_num, input_content, expected_output
                        )
                        test_code_file_path = f"tests/test_{problem_id}.py"
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
    input_content_escaped = input_content.replace("'", "\\'").replace("\n", "\\n")
    expected_output_escaped = expected_output.replace("'", "\\'").replace("\n", "\\n")
    return (
        f"def test_problem_{problem_id}_{input_num}():\n"
        f"    input_content = '{input_content_escaped}'\n"
        f"    expected_output = '{expected_output_escaped}'\n"
        f"    run_pie_test_case('../{problem_id}.py', input_content, expected_output)\n"
    )


if __name__ == "__main__":
    create_files()
