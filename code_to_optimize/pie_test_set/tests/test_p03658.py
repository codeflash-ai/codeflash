from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03658_0():
    input_content = "5 3\n1 2 3 4 5"
    expected_output = "12"
    run_pie_test_case("../p03658.py", input_content, expected_output)


def test_problem_p03658_1():
    input_content = "15 14\n50 26 27 21 41 7 42 35 7 5 5 36 39 1 45"
    expected_output = "386"
    run_pie_test_case("../p03658.py", input_content, expected_output)


def test_problem_p03658_2():
    input_content = "5 3\n1 2 3 4 5"
    expected_output = "12"
    run_pie_test_case("../p03658.py", input_content, expected_output)
