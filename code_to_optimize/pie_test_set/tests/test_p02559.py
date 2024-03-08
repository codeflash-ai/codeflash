from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02559_0():
    input_content = "5 5\n1 2 3 4 5\n1 0 5\n1 2 4\n0 3 10\n1 0 5\n1 0 3"
    expected_output = "15\n7\n25\n6"
    run_pie_test_case("../p02559.py", input_content, expected_output)


def test_problem_p02559_1():
    input_content = "5 5\n1 2 3 4 5\n1 0 5\n1 2 4\n0 3 10\n1 0 5\n1 0 3"
    expected_output = "15\n7\n25\n6"
    run_pie_test_case("../p02559.py", input_content, expected_output)
