from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00966_0():
    input_content = "9 4 5 4\n3 C\n4 I\n7 C\n9 P\n2 1\n4 0\n6 2\n7 0\n8 4\n8\n1\n9\n6"
    expected_output = "ICPC"
    run_pie_test_case("../p00966.py", input_content, expected_output)


def test_problem_p00966_1():
    input_content = "9 4 5 4\n3 C\n4 I\n7 C\n9 P\n2 1\n4 0\n6 2\n7 0\n8 4\n8\n1\n9\n6"
    expected_output = "ICPC"
    run_pie_test_case("../p00966.py", input_content, expected_output)
