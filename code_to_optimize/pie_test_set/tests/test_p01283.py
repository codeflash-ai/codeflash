from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01283_0():
    input_content = "5\n5 4 3 2 1\n5\n7 7 7 7 7\n10\n186 8 42 24 154 40 10 56 122 72\n0"
    expected_output = "0 1 1\n0 0 0\n8 7 14"
    run_pie_test_case("../p01283.py", input_content, expected_output)


def test_problem_p01283_1():
    input_content = "5\n5 4 3 2 1\n5\n7 7 7 7 7\n10\n186 8 42 24 154 40 10 56 122 72\n0"
    expected_output = "0 1 1\n0 0 0\n8 7 14"
    run_pie_test_case("../p01283.py", input_content, expected_output)
