from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03334_0():
    input_content = "2 1 2"
    expected_output = "0 0\n0 2\n2 0\n2 2"
    run_pie_test_case("../p03334.py", input_content, expected_output)


def test_problem_p03334_1():
    input_content = "3 1 5"
    expected_output = "0 0\n0 2\n0 4\n1 1\n1 3\n1 5\n2 0\n2 2\n2 4"
    run_pie_test_case("../p03334.py", input_content, expected_output)


def test_problem_p03334_2():
    input_content = "2 1 2"
    expected_output = "0 0\n0 2\n2 0\n2 2"
    run_pie_test_case("../p03334.py", input_content, expected_output)
