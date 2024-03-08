from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03677_0():
    input_content = "4 6\n1 5 1 4"
    expected_output = "5"
    run_pie_test_case("../p03677.py", input_content, expected_output)


def test_problem_p03677_1():
    input_content = "4 6\n1 5 1 4"
    expected_output = "5"
    run_pie_test_case("../p03677.py", input_content, expected_output)


def test_problem_p03677_2():
    input_content = "10 10\n10 9 8 7 6 5 4 3 2 1"
    expected_output = "45"
    run_pie_test_case("../p03677.py", input_content, expected_output)
