from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02540_0():
    input_content = "4\n1 4\n2 3\n3 1\n4 2"
    expected_output = "1\n1\n2\n2"
    run_pie_test_case("../p02540.py", input_content, expected_output)


def test_problem_p02540_1():
    input_content = "7\n6 4\n4 3\n3 5\n7 1\n2 7\n5 2\n1 6"
    expected_output = "3\n3\n1\n1\n2\n3\n2"
    run_pie_test_case("../p02540.py", input_content, expected_output)


def test_problem_p02540_2():
    input_content = "4\n1 4\n2 3\n3 1\n4 2"
    expected_output = "1\n1\n2\n2"
    run_pie_test_case("../p02540.py", input_content, expected_output)
