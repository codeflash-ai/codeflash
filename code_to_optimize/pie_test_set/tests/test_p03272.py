from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03272_0():
    input_content = "4 2"
    expected_output = "3"
    run_pie_test_case("../p03272.py", input_content, expected_output)


def test_problem_p03272_1():
    input_content = "1 1"
    expected_output = "1"
    run_pie_test_case("../p03272.py", input_content, expected_output)


def test_problem_p03272_2():
    input_content = "4 2"
    expected_output = "3"
    run_pie_test_case("../p03272.py", input_content, expected_output)


def test_problem_p03272_3():
    input_content = "15 11"
    expected_output = "5"
    run_pie_test_case("../p03272.py", input_content, expected_output)
