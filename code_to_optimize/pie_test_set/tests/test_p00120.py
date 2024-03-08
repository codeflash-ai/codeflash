from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00120_0():
    input_content = "30 4 5 6\n30 5 5 5\n50 3 3 3 10 10\n49 3 3 3 10 10"
    expected_output = "OK\nOK\nOK\nNA"
    run_pie_test_case("../p00120.py", input_content, expected_output)


def test_problem_p00120_1():
    input_content = "30 4 5 6\n30 5 5 5\n50 3 3 3 10 10\n49 3 3 3 10 10"
    expected_output = "OK\nOK\nOK\nNA"
    run_pie_test_case("../p00120.py", input_content, expected_output)
