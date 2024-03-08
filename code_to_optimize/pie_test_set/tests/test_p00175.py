from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00175_0():
    input_content = "7\n4\n0\n12\n10\n10000\n-1"
    expected_output = "13\n10\n0\n30\n22\n2130100"
    run_pie_test_case("../p00175.py", input_content, expected_output)


def test_problem_p00175_1():
    input_content = "7\n4\n0\n12\n10\n10000\n-1"
    expected_output = "13\n10\n0\n30\n22\n2130100"
    run_pie_test_case("../p00175.py", input_content, expected_output)
