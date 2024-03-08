from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03109_0():
    input_content = "2019/04/30"
    expected_output = "Heisei"
    run_pie_test_case("../p03109.py", input_content, expected_output)


def test_problem_p03109_1():
    input_content = "2019/04/30"
    expected_output = "Heisei"
    run_pie_test_case("../p03109.py", input_content, expected_output)


def test_problem_p03109_2():
    input_content = "2019/11/01"
    expected_output = "TBD"
    run_pie_test_case("../p03109.py", input_content, expected_output)
