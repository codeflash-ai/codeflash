from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00074_0():
    input_content = "1 30 0\n-1 -1 -1"
    expected_output = "00:30:00\n01:30:00"
    run_pie_test_case("../p00074.py", input_content, expected_output)


def test_problem_p00074_1():
    input_content = "1 30 0\n-1 -1 -1"
    expected_output = "00:30:00\n01:30:00"
    run_pie_test_case("../p00074.py", input_content, expected_output)
