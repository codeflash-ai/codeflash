from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00461_0():
    input_content = "1\n13\nOOIOIOIOIIOII\n2\n13\nOOIOIOIOIIOII\n0"
    expected_output = "4\n2"
    run_pie_test_case("../p00461.py", input_content, expected_output)


def test_problem_p00461_1():
    input_content = "1\n13\nOOIOIOIOIIOII\n2\n13\nOOIOIOIOIIOII\n0"
    expected_output = "4\n2"
    run_pie_test_case("../p00461.py", input_content, expected_output)
