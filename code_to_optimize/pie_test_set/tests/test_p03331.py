from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03331_0():
    input_content = "15"
    expected_output = "6"
    run_pie_test_case("../p03331.py", input_content, expected_output)


def test_problem_p03331_1():
    input_content = "15"
    expected_output = "6"
    run_pie_test_case("../p03331.py", input_content, expected_output)


def test_problem_p03331_2():
    input_content = "100000"
    expected_output = "10"
    run_pie_test_case("../p03331.py", input_content, expected_output)
