from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03970_0():
    input_content = "C0DEFESTIVAL2O16"
    expected_output = "2"
    run_pie_test_case("../p03970.py", input_content, expected_output)


def test_problem_p03970_1():
    input_content = "C0DEFESTIVAL2O16"
    expected_output = "2"
    run_pie_test_case("../p03970.py", input_content, expected_output)


def test_problem_p03970_2():
    input_content = "FESTIVAL2016CODE"
    expected_output = "16"
    run_pie_test_case("../p03970.py", input_content, expected_output)
