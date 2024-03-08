from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02698_0():
    input_content = "10\n1 2 5 3 4 6 7 3 2 4\n1 2\n2 3\n3 4\n4 5\n3 6\n6 7\n1 8\n8 9\n9 10"
    expected_output = "1\n2\n3\n3\n4\n4\n5\n2\n2\n3"
    run_pie_test_case("../p02698.py", input_content, expected_output)


def test_problem_p02698_1():
    input_content = "10\n1 2 5 3 4 6 7 3 2 4\n1 2\n2 3\n3 4\n4 5\n3 6\n6 7\n1 8\n8 9\n9 10"
    expected_output = "1\n2\n3\n3\n4\n4\n5\n2\n2\n3"
    run_pie_test_case("../p02698.py", input_content, expected_output)
