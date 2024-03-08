from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03438_0():
    input_content = "3\n1 2 3\n5 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03438.py", input_content, expected_output)


def test_problem_p03438_1():
    input_content = "5\n3 1 4 1 5\n2 7 1 8 2"
    expected_output = "No"
    run_pie_test_case("../p03438.py", input_content, expected_output)


def test_problem_p03438_2():
    input_content = "5\n2 7 1 8 2\n3 1 4 1 5"
    expected_output = "No"
    run_pie_test_case("../p03438.py", input_content, expected_output)


def test_problem_p03438_3():
    input_content = "3\n1 2 3\n5 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03438.py", input_content, expected_output)
