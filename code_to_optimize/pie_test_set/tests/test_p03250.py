from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03250_0():
    input_content = "1 5 2"
    expected_output = "53"
    run_pie_test_case("../p03250.py", input_content, expected_output)


def test_problem_p03250_1():
    input_content = "9 9 9"
    expected_output = "108"
    run_pie_test_case("../p03250.py", input_content, expected_output)


def test_problem_p03250_2():
    input_content = "1 5 2"
    expected_output = "53"
    run_pie_test_case("../p03250.py", input_content, expected_output)


def test_problem_p03250_3():
    input_content = "6 6 7"
    expected_output = "82"
    run_pie_test_case("../p03250.py", input_content, expected_output)
