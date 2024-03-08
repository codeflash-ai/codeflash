from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00740_0():
    input_content = "3 2\n3 3\n3 50\n10 29\n31 32\n50 2\n50 50\n0 0"
    expected_output = "1\n0\n1\n5\n30\n1\n13"
    run_pie_test_case("../p00740.py", input_content, expected_output)


def test_problem_p00740_1():
    input_content = "3 2\n3 3\n3 50\n10 29\n31 32\n50 2\n50 50\n0 0"
    expected_output = "1\n0\n1\n5\n30\n1\n13"
    run_pie_test_case("../p00740.py", input_content, expected_output)
