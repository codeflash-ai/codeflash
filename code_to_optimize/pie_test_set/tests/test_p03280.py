from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03280_0():
    input_content = "2 2"
    expected_output = "1"
    run_pie_test_case("../p03280.py", input_content, expected_output)


def test_problem_p03280_1():
    input_content = "5 7"
    expected_output = "24"
    run_pie_test_case("../p03280.py", input_content, expected_output)


def test_problem_p03280_2():
    input_content = "2 2"
    expected_output = "1"
    run_pie_test_case("../p03280.py", input_content, expected_output)
