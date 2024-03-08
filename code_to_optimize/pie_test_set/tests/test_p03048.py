from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03048_0():
    input_content = "1 2 3 4"
    expected_output = "4"
    run_pie_test_case("../p03048.py", input_content, expected_output)


def test_problem_p03048_1():
    input_content = "1 2 3 4"
    expected_output = "4"
    run_pie_test_case("../p03048.py", input_content, expected_output)


def test_problem_p03048_2():
    input_content = "13 1 4 3000"
    expected_output = "87058"
    run_pie_test_case("../p03048.py", input_content, expected_output)
