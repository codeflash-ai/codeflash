from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03186_0():
    input_content = "3 1 4"
    expected_output = "5"
    run_pie_test_case("../p03186.py", input_content, expected_output)


def test_problem_p03186_1():
    input_content = "3 1 4"
    expected_output = "5"
    run_pie_test_case("../p03186.py", input_content, expected_output)


def test_problem_p03186_2():
    input_content = "8 8 1"
    expected_output = "9"
    run_pie_test_case("../p03186.py", input_content, expected_output)


def test_problem_p03186_3():
    input_content = "5 2 9"
    expected_output = "10"
    run_pie_test_case("../p03186.py", input_content, expected_output)
