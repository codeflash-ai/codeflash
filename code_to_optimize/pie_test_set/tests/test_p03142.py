from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03142_0():
    input_content = "3 1\n1 2\n1 3\n2 3"
    expected_output = "0\n1\n2"
    run_pie_test_case("../p03142.py", input_content, expected_output)


def test_problem_p03142_1():
    input_content = "6 3\n2 1\n2 3\n4 1\n4 2\n6 1\n2 6\n4 6\n6 5"
    expected_output = "6\n4\n2\n0\n6\n2"
    run_pie_test_case("../p03142.py", input_content, expected_output)


def test_problem_p03142_2():
    input_content = "3 1\n1 2\n1 3\n2 3"
    expected_output = "0\n1\n2"
    run_pie_test_case("../p03142.py", input_content, expected_output)
