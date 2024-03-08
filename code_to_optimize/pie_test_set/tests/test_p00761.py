from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00761_0():
    input_content = "2012 4\n83268 6\n1112 4\n0 1\n99 2\n0 0"
    expected_output = "3 6174 1\n1 862632 7\n5 6174 1\n0 0 1\n1 0 1"
    run_pie_test_case("../p00761.py", input_content, expected_output)


def test_problem_p00761_1():
    input_content = "2012 4\n83268 6\n1112 4\n0 1\n99 2\n0 0"
    expected_output = "3 6174 1\n1 862632 7\n5 6174 1\n0 0 1\n1 0 1"
    run_pie_test_case("../p00761.py", input_content, expected_output)
