from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03088_0():
    input_content = "3"
    expected_output = "61"
    run_pie_test_case("../p03088.py", input_content, expected_output)


def test_problem_p03088_1():
    input_content = "4"
    expected_output = "230"
    run_pie_test_case("../p03088.py", input_content, expected_output)


def test_problem_p03088_2():
    input_content = "3"
    expected_output = "61"
    run_pie_test_case("../p03088.py", input_content, expected_output)


def test_problem_p03088_3():
    input_content = "100"
    expected_output = "388130742"
    run_pie_test_case("../p03088.py", input_content, expected_output)
