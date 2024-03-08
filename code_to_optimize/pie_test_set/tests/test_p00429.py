from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00429_0():
    input_content = "5\n11\n5\n11\n0"
    expected_output = "13112221\n13112221"
    run_pie_test_case("../p00429.py", input_content, expected_output)


def test_problem_p00429_1():
    input_content = "5\n11\n5\n11\n0"
    expected_output = "13112221\n13112221"
    run_pie_test_case("../p00429.py", input_content, expected_output)
