from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03055_0():
    input_content = "3\n1 2\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03055.py", input_content, expected_output)


def test_problem_p03055_1():
    input_content = "7\n1 7\n7 4\n3 4\n7 5\n6 3\n2 1"
    expected_output = "First"
    run_pie_test_case("../p03055.py", input_content, expected_output)


def test_problem_p03055_2():
    input_content = "3\n1 2\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03055.py", input_content, expected_output)


def test_problem_p03055_3():
    input_content = "6\n1 2\n2 3\n2 4\n4 6\n5 6"
    expected_output = "Second"
    run_pie_test_case("../p03055.py", input_content, expected_output)
