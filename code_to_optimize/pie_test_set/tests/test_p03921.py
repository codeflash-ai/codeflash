from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03921_0():
    input_content = "4 6\n3 1 2 3\n2 4 2\n2 4 6\n1 6"
    expected_output = "YES"
    run_pie_test_case("../p03921.py", input_content, expected_output)


def test_problem_p03921_1():
    input_content = "4 6\n3 1 2 3\n2 4 2\n2 4 6\n1 6"
    expected_output = "YES"
    run_pie_test_case("../p03921.py", input_content, expected_output)


def test_problem_p03921_2():
    input_content = "4 4\n2 1 2\n2 1 2\n1 3\n2 4 3"
    expected_output = "NO"
    run_pie_test_case("../p03921.py", input_content, expected_output)
