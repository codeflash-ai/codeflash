from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00005_0():
    input_content = "8 6\n50000000 30000000"
    expected_output = "2 24\n10000000 150000000"
    run_pie_test_case("../p00005.py", input_content, expected_output)


def test_problem_p00005_1():
    input_content = "8 6\n50000000 30000000"
    expected_output = "2 24\n10000000 150000000"
    run_pie_test_case("../p00005.py", input_content, expected_output)
