from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02370_0():
    input_content = "6 6\n0 1\n1 2\n3 1\n3 4\n4 5\n5 2"
    expected_output = "0\n3\n1\n4\n5\n2"
    run_pie_test_case("../p02370.py", input_content, expected_output)


def test_problem_p02370_1():
    input_content = "6 6\n0 1\n1 2\n3 1\n3 4\n4 5\n5 2"
    expected_output = "0\n3\n1\n4\n5\n2"
    run_pie_test_case("../p02370.py", input_content, expected_output)
