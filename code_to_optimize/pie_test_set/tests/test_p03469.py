from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03469_0():
    input_content = "2017/01/07"
    expected_output = "2018/01/07"
    run_pie_test_case("../p03469.py", input_content, expected_output)


def test_problem_p03469_1():
    input_content = "2017/01/07"
    expected_output = "2018/01/07"
    run_pie_test_case("../p03469.py", input_content, expected_output)


def test_problem_p03469_2():
    input_content = "2017/01/31"
    expected_output = "2018/01/31"
    run_pie_test_case("../p03469.py", input_content, expected_output)
