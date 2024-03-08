from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03726_0():
    input_content = "3\n1 2\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03726.py", input_content, expected_output)


def test_problem_p03726_1():
    input_content = "3\n1 2\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03726.py", input_content, expected_output)


def test_problem_p03726_2():
    input_content = "6\n1 2\n2 3\n3 4\n2 5\n5 6"
    expected_output = "Second"
    run_pie_test_case("../p03726.py", input_content, expected_output)


def test_problem_p03726_3():
    input_content = "4\n1 2\n2 3\n2 4"
    expected_output = "First"
    run_pie_test_case("../p03726.py", input_content, expected_output)
