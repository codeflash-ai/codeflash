from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03213_0():
    input_content = "9"
    expected_output = "0"
    run_pie_test_case("../p03213.py", input_content, expected_output)


def test_problem_p03213_1():
    input_content = "9"
    expected_output = "0"
    run_pie_test_case("../p03213.py", input_content, expected_output)


def test_problem_p03213_2():
    input_content = "100"
    expected_output = "543"
    run_pie_test_case("../p03213.py", input_content, expected_output)


def test_problem_p03213_3():
    input_content = "10"
    expected_output = "1"
    run_pie_test_case("../p03213.py", input_content, expected_output)
