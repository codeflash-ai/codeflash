from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00016_0():
    input_content = "56,65\n97,54\n64,-4\n55,76\n42,-27\n43,80\n87,-86\n55,-6\n89,34\n95,5\n0,0"
    expected_output = "171\n-302"
    run_pie_test_case("../p00016.py", input_content, expected_output)


def test_problem_p00016_1():
    input_content = "56,65\n97,54\n64,-4\n55,76\n42,-27\n43,80\n87,-86\n55,-6\n89,34\n95,5\n0,0"
    expected_output = "171\n-302"
    run_pie_test_case("../p00016.py", input_content, expected_output)
