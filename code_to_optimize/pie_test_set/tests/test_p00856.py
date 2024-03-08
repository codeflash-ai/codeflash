from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00856_0():
    input_content = "6 1 0 0\n7 1 0 0\n7 2 0 0\n6 6 1 1\n2\n5\n7 10 0 6\n1\n2\n3\n4\n5\n6\n0 0 0 0"
    expected_output = "0.166667\n0.000000\n0.166667\n0.619642\n0.000000"
    run_pie_test_case("../p00856.py", input_content, expected_output)


def test_problem_p00856_1():
    input_content = "6 1 0 0\n7 1 0 0\n7 2 0 0\n6 6 1 1\n2\n5\n7 10 0 6\n1\n2\n3\n4\n5\n6\n0 0 0 0"
    expected_output = "0.166667\n0.000000\n0.166667\n0.619642\n0.000000"
    run_pie_test_case("../p00856.py", input_content, expected_output)
