from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00174_0():
    input_content = "ABAABBBAABABAAABBAA\nAABBBABBABBAAABABABAAB\nBABAABAABABABBAAAB\nAABABAAABBAABBBABAA\nAAAAAAAAAAA\nABBBBBBBBBB\n0"
    expected_output = "11 8\n10 12\n11 7\n11 8\n11 0\n0 11"
    run_pie_test_case("../p00174.py", input_content, expected_output)


def test_problem_p00174_1():
    input_content = "ABAABBBAABABAAABBAA\nAABBBABBABBAAABABABAAB\nBABAABAABABABBAAAB\nAABABAAABBAABBBABAA\nAAAAAAAAAAA\nABBBBBBBBBB\n0"
    expected_output = "11 8\n10 12\n11 7\n11 8\n11 0\n0 11"
    run_pie_test_case("../p00174.py", input_content, expected_output)
