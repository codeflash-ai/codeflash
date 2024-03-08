from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04049_0():
    input_content = "6 2\n1 2\n3 2\n4 2\n1 6\n5 6"
    expected_output = "2"
    run_pie_test_case("../p04049.py", input_content, expected_output)


def test_problem_p04049_1():
    input_content = "6 2\n1 2\n3 2\n4 2\n1 6\n5 6"
    expected_output = "2"
    run_pie_test_case("../p04049.py", input_content, expected_output)


def test_problem_p04049_2():
    input_content = "6 5\n1 2\n3 2\n4 2\n1 6\n5 6"
    expected_output = "0"
    run_pie_test_case("../p04049.py", input_content, expected_output)
