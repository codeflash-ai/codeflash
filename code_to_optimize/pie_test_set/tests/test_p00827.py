from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00827_0():
    input_content = "700 300 200\n500 200 300\n500 200 500\n275 110 330\n275 110 385\n648 375 4002\n3 1 10000\n0 0 0"
    expected_output = "1 3\n1 1\n1 0\n0 3\n1 1\n49 74\n3333 1"
    run_pie_test_case("../p00827.py", input_content, expected_output)


def test_problem_p00827_1():
    input_content = "700 300 200\n500 200 300\n500 200 500\n275 110 330\n275 110 385\n648 375 4002\n3 1 10000\n0 0 0"
    expected_output = "1 3\n1 1\n1 0\n0 3\n1 1\n49 74\n3333 1"
    run_pie_test_case("../p00827.py", input_content, expected_output)
