from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00046_0():
    input_content = "3776.0\n1819.0\n645.2\n2004.1\n1208.6"
    expected_output = "3130.8"
    run_pie_test_case("../p00046.py", input_content, expected_output)


def test_problem_p00046_1():
    input_content = "3776.0\n1819.0\n645.2\n2004.1\n1208.6"
    expected_output = "3130.8"
    run_pie_test_case("../p00046.py", input_content, expected_output)
