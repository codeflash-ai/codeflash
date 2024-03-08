from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00108_0():
    input_content = "10\n4 5 1 1 4 5 12 3 5 4\n0"
    expected_output = "3\n6 6 4 4 6 6 4 4 6 6"
    run_pie_test_case("../p00108.py", input_content, expected_output)


def test_problem_p00108_1():
    input_content = "10\n4 5 1 1 4 5 12 3 5 4\n0"
    expected_output = "3\n6 6 4 4 6 6 4 4 6 6"
    run_pie_test_case("../p00108.py", input_content, expected_output)
