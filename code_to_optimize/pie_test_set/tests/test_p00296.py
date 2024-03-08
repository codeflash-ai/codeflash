from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00296_0():
    input_content = "10 5 3\n2 6 5 18 3\n3 0 5"
    expected_output = "1\n0\n1"
    run_pie_test_case("../p00296.py", input_content, expected_output)
