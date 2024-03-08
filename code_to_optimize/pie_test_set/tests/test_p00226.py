from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00226_0():
    input_content = "1234 5678\n1234 1354\n1234 1234\n1230 1023\n0123 1234\n0 0"
    expected_output = "0 0\n2 1\n4 0\n1 3\n0 3"
    run_pie_test_case("../p00226.py", input_content, expected_output)


def test_problem_p00226_1():
    input_content = "1234 5678\n1234 1354\n1234 1234\n1230 1023\n0123 1234\n0 0"
    expected_output = "0 0\n2 1\n4 0\n1 3\n0 3"
    run_pie_test_case("../p00226.py", input_content, expected_output)
