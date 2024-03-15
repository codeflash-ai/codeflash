import os
import subprocess


def run_pie_test_case(script_path, test_input, expected_output):
    assert os.path.exists(script_path), f"Script file does not exist: {script_path}"
    process = subprocess.Popen(
        ["python", script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(f"Running command: python {script_path}")
    stdout, stderr = process.communicate(input=test_input)
    assert process.returncode == 0, f"Process exited with code {process.returncode}"
    if stderr:
        print(f"Error in stderr: {stderr}")
        assert False, f"Script error: {stderr}"
    assert (
        stdout.strip() == expected_output
    ), f"Expected '{expected_output}' but got '{stdout.strip()}'"
