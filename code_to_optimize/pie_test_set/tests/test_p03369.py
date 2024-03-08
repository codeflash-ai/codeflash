from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03369_0():
    input_content = "oxo"
    expected_output = "900"
    run_pie_test_case("../p03369.py", input_content, expected_output)


def test_problem_p03369_1():
    input_content = "oxo"
    expected_output = "900"
    run_pie_test_case("../p03369.py", input_content, expected_output)


def test_problem_p03369_2():
    input_content = "xxx"
    expected_output = "700"
    run_pie_test_case("../p03369.py", input_content, expected_output)


def test_problem_p03369_3():
    input_content = "ooo"
    expected_output = "1000"
    run_pie_test_case("../p03369.py", input_content, expected_output)
