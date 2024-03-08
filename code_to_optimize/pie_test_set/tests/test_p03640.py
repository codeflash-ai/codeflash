from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03640_0():
    input_content = "2 2\n3\n2 1 1"
    expected_output = "1 1\n2 3"
    run_pie_test_case("../p03640.py", input_content, expected_output)


def test_problem_p03640_1():
    input_content = "1 1\n1\n1"
    expected_output = "1"
    run_pie_test_case("../p03640.py", input_content, expected_output)


def test_problem_p03640_2():
    input_content = "3 5\n5\n1 2 3 4 5"
    expected_output = "1 4 4 4 3\n2 5 4 5 3\n2 5 5 5 3"
    run_pie_test_case("../p03640.py", input_content, expected_output)


def test_problem_p03640_3():
    input_content = "2 2\n3\n2 1 1"
    expected_output = "1 1\n2 3"
    run_pie_test_case("../p03640.py", input_content, expected_output)
