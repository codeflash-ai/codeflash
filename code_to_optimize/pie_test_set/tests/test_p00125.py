from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00125_0():
    input_content = "2006 9 2 2006 9 3\n2006 9 2 2006 11 11\n2004 1 1 2005 1 1\n2000 1 1 2006 1 1\n2000 1 1 2101 1 1\n-1 -1 -1 -1 -1 -1"
    expected_output = "1\n70\n366\n2192\n36890"
    run_pie_test_case("../p00125.py", input_content, expected_output)


def test_problem_p00125_1():
    input_content = "2006 9 2 2006 9 3\n2006 9 2 2006 11 11\n2004 1 1 2005 1 1\n2000 1 1 2006 1 1\n2000 1 1 2101 1 1\n-1 -1 -1 -1 -1 -1"
    expected_output = "1\n70\n366\n2192\n36890"
    run_pie_test_case("../p00125.py", input_content, expected_output)
