from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00062_0():
    input_content = "4823108376\n1234567890\n0123456789"
    expected_output = "5\n6\n4"
    run_pie_test_case("../p00062.py", input_content, expected_output)


def test_problem_p00062_1():
    input_content = "4823108376\n1234567890\n0123456789"
    expected_output = "5\n6\n4"
    run_pie_test_case("../p00062.py", input_content, expected_output)
