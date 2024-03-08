from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03130_0():
    input_content = "4 2\n1 3\n2 3"
    expected_output = "YES"
    run_pie_test_case("../p03130.py", input_content, expected_output)


def test_problem_p03130_1():
    input_content = "4 2\n1 3\n2 3"
    expected_output = "YES"
    run_pie_test_case("../p03130.py", input_content, expected_output)


def test_problem_p03130_2():
    input_content = "2 1\n3 2\n4 3"
    expected_output = "YES"
    run_pie_test_case("../p03130.py", input_content, expected_output)


def test_problem_p03130_3():
    input_content = "3 2\n2 4\n1 2"
    expected_output = "NO"
    run_pie_test_case("../p03130.py", input_content, expected_output)
