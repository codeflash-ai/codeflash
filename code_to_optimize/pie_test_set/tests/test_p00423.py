from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00423_0():
    input_content = "3\n9 1\n5 4\n0 8\n3\n9 1\n5 4\n1 0\n3\n9 1\n5 5\n1 8\n0"
    expected_output = "19 8\n20 0\n15 14"
    run_pie_test_case("../p00423.py", input_content, expected_output)


def test_problem_p00423_1():
    input_content = "3\n9 1\n5 4\n0 8\n3\n9 1\n5 4\n1 0\n3\n9 1\n5 5\n1 8\n0"
    expected_output = "19 8\n20 0\n15 14"
    run_pie_test_case("../p00423.py", input_content, expected_output)
