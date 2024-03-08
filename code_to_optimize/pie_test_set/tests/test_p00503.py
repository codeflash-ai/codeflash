from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00503_0():
    input_content = "3 2\n30 50 0 50 70 100\n10 20 20 70 90 60\n40 60 20 90 90 70"
    expected_output = "49000"
    run_pie_test_case("../p00503.py", input_content, expected_output)


def test_problem_p00503_1():
    input_content = "3 2\n30 50 0 50 70 100\n10 20 20 70 90 60\n40 60 20 90 90 70"
    expected_output = "49000"
    run_pie_test_case("../p00503.py", input_content, expected_output)
