from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00149_0():
    input_content = "1.0 1.2\n0.8 1.5\n1.2 0.7\n2.0 2.0"
    expected_output = "2 3\n2 1\n0 0\n0 0"
    run_pie_test_case("../p00149.py", input_content, expected_output)


def test_problem_p00149_1():
    input_content = "1.0 1.2\n0.8 1.5\n1.2 0.7\n2.0 2.0"
    expected_output = "2 3\n2 1\n0 0\n0 0"
    run_pie_test_case("../p00149.py", input_content, expected_output)
