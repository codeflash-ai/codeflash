from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02373_0():
    input_content = "8\n3 1 2 3\n2 4 5\n0\n0\n0\n2 6 7\n0\n0\n4\n4 6\n4 7\n4 3\n5 2"
    expected_output = "1\n1\n0\n0"
    run_pie_test_case("../p02373.py", input_content, expected_output)


def test_problem_p02373_1():
    input_content = "8\n3 1 2 3\n2 4 5\n0\n0\n0\n2 6 7\n0\n0\n4\n4 6\n4 7\n4 3\n5 2"
    expected_output = "1\n1\n0\n0"
    run_pie_test_case("../p02373.py", input_content, expected_output)
