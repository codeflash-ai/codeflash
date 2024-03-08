from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02297_0():
    input_content = "3\n0 0\n2 2\n-1 1"
    expected_output = "2.0"
    run_pie_test_case("../p02297.py", input_content, expected_output)


def test_problem_p02297_1():
    input_content = "4\n0 0\n1 1\n1 2\n0 2"
    expected_output = "1.5"
    run_pie_test_case("../p02297.py", input_content, expected_output)


def test_problem_p02297_2():
    input_content = "3\n0 0\n2 2\n-1 1"
    expected_output = "2.0"
    run_pie_test_case("../p02297.py", input_content, expected_output)
