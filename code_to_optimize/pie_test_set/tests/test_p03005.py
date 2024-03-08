from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03005_0():
    input_content = "3 2"
    expected_output = "1"
    run_pie_test_case("../p03005.py", input_content, expected_output)


def test_problem_p03005_1():
    input_content = "8 5"
    expected_output = "3"
    run_pie_test_case("../p03005.py", input_content, expected_output)


def test_problem_p03005_2():
    input_content = "3 1"
    expected_output = "0"
    run_pie_test_case("../p03005.py", input_content, expected_output)


def test_problem_p03005_3():
    input_content = "3 2"
    expected_output = "1"
    run_pie_test_case("../p03005.py", input_content, expected_output)
