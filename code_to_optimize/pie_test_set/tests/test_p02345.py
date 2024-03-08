from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02345_0():
    input_content = "3 5\n0 0 1\n0 1 2\n0 2 3\n1 0 2\n1 1 2"
    expected_output = "1\n2"
    run_pie_test_case("../p02345.py", input_content, expected_output)


def test_problem_p02345_1():
    input_content = "3 5\n0 0 1\n0 1 2\n0 2 3\n1 0 2\n1 1 2"
    expected_output = "1\n2"
    run_pie_test_case("../p02345.py", input_content, expected_output)


def test_problem_p02345_2():
    input_content = "1 3\n1 0 0\n0 0 5\n1 0 0"
    expected_output = "2147483647\n5"
    run_pie_test_case("../p02345.py", input_content, expected_output)
