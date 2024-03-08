from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00017_0():
    input_content = "xlmw mw xli tmgxyvi xlex m xsso mr xli xvmt."
    expected_output = "this is the picture that i took in the trip."
    run_pie_test_case("../p00017.py", input_content, expected_output)


def test_problem_p00017_1():
    input_content = "xlmw mw xli tmgxyvi xlex m xsso mr xli xvmt."
    expected_output = "this is the picture that i took in the trip."
    run_pie_test_case("../p00017.py", input_content, expected_output)
