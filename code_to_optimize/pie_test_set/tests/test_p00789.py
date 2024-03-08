from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00789_0():
    input_content = "2\n10\n30\n0"
    expected_output = "1\n4\n27"
    run_pie_test_case("../p00789.py", input_content, expected_output)


def test_problem_p00789_1():
    input_content = "2\n10\n30\n0"
    expected_output = "1\n4\n27"
    run_pie_test_case("../p00789.py", input_content, expected_output)
