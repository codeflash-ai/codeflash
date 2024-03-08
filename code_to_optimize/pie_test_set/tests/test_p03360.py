from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03360_0():
    input_content = "5 3 11\n1"
    expected_output = "30"
    run_pie_test_case("../p03360.py", input_content, expected_output)


def test_problem_p03360_1():
    input_content = "5 3 11\n1"
    expected_output = "30"
    run_pie_test_case("../p03360.py", input_content, expected_output)


def test_problem_p03360_2():
    input_content = "3 3 4\n2"
    expected_output = "22"
    run_pie_test_case("../p03360.py", input_content, expected_output)
