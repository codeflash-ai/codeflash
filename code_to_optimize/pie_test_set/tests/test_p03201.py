from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03201_0():
    input_content = "3\n1 2 3"
    expected_output = "1"
    run_pie_test_case("../p03201.py", input_content, expected_output)


def test_problem_p03201_1():
    input_content = "3\n1 2 3"
    expected_output = "1"
    run_pie_test_case("../p03201.py", input_content, expected_output)


def test_problem_p03201_2():
    input_content = "5\n3 11 14 5 13"
    expected_output = "2"
    run_pie_test_case("../p03201.py", input_content, expected_output)
