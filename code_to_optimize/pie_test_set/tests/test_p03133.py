from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03133_0():
    input_content = "2 2\n0 1\n1 0"
    expected_output = "6"
    run_pie_test_case("../p03133.py", input_content, expected_output)


def test_problem_p03133_1():
    input_content = "2 2\n0 1\n1 0"
    expected_output = "6"
    run_pie_test_case("../p03133.py", input_content, expected_output)


def test_problem_p03133_2():
    input_content = "2 3\n0 0 0\n0 1 0"
    expected_output = "8"
    run_pie_test_case("../p03133.py", input_content, expected_output)
