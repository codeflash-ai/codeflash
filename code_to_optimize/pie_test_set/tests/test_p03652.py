from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03652_0():
    input_content = "4 5\n5 1 3 4 2\n2 5 3 1 4\n2 3 1 4 5\n2 5 4 3 1"
    expected_output = "2"
    run_pie_test_case("../p03652.py", input_content, expected_output)


def test_problem_p03652_1():
    input_content = "3 3\n2 1 3\n2 1 3\n2 1 3"
    expected_output = "3"
    run_pie_test_case("../p03652.py", input_content, expected_output)


def test_problem_p03652_2():
    input_content = "4 5\n5 1 3 4 2\n2 5 3 1 4\n2 3 1 4 5\n2 5 4 3 1"
    expected_output = "2"
    run_pie_test_case("../p03652.py", input_content, expected_output)
