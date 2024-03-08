from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00244_0():
    input_content = "2 1\n1 2 5\n3 2\n1 2 5\n2 3 5\n6 9\n1 2 7\n1 3 9\n1 5 14\n2 3 10\n2 4 15\n3 4 11\n3 5 2\n4 5 9\n4 6 8\n0 0"
    expected_output = "5\n0\n7"
    run_pie_test_case("../p00244.py", input_content, expected_output)


def test_problem_p00244_1():
    input_content = "2 1\n1 2 5\n3 2\n1 2 5\n2 3 5\n6 9\n1 2 7\n1 3 9\n1 5 14\n2 3 10\n2 4 15\n3 4 11\n3 5 2\n4 5 9\n4 6 8\n0 0"
    expected_output = "5\n0\n7"
    run_pie_test_case("../p00244.py", input_content, expected_output)
