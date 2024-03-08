from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03465_0():
    input_content = "3\n1 2 1"
    expected_output = "2"
    run_pie_test_case("../p03465.py", input_content, expected_output)


def test_problem_p03465_1():
    input_content = "3\n1 2 1"
    expected_output = "2"
    run_pie_test_case("../p03465.py", input_content, expected_output)


def test_problem_p03465_2():
    input_content = "1\n58"
    expected_output = "58"
    run_pie_test_case("../p03465.py", input_content, expected_output)
