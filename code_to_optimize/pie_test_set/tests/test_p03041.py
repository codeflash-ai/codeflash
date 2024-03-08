from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03041_0():
    input_content = "3 1\nABC"
    expected_output = "aBC"
    run_pie_test_case("../p03041.py", input_content, expected_output)


def test_problem_p03041_1():
    input_content = "4 3\nCABA"
    expected_output = "CAbA"
    run_pie_test_case("../p03041.py", input_content, expected_output)


def test_problem_p03041_2():
    input_content = "3 1\nABC"
    expected_output = "aBC"
    run_pie_test_case("../p03041.py", input_content, expected_output)
