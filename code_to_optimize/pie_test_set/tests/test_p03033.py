from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03033_0():
    input_content = "4 6\n1 3 2\n7 13 10\n18 20 13\n3 4 2\n0\n1\n2\n3\n5\n8"
    expected_output = "2\n2\n10\n-1\n13\n-1"
    run_pie_test_case("../p03033.py", input_content, expected_output)


def test_problem_p03033_1():
    input_content = "4 6\n1 3 2\n7 13 10\n18 20 13\n3 4 2\n0\n1\n2\n3\n5\n8"
    expected_output = "2\n2\n10\n-1\n13\n-1"
    run_pie_test_case("../p03033.py", input_content, expected_output)
