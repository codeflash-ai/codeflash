from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00093_0():
    input_content = "2001 2010\n2005 2005\n2001 2010\n0 0"
    expected_output = "2004\n2008\n\nNA\n\n2004\n2008"
    run_pie_test_case("../p00093.py", input_content, expected_output)


def test_problem_p00093_1():
    input_content = "2001 2010\n2005 2005\n2001 2010\n0 0"
    expected_output = "2004\n2008\n\nNA\n\n2004\n2008"
    run_pie_test_case("../p00093.py", input_content, expected_output)
