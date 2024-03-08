from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03787_0():
    input_content = "3 1\n1 2"
    expected_output = "7"
    run_pie_test_case("../p03787.py", input_content, expected_output)


def test_problem_p03787_1():
    input_content = "7 5\n1 2\n3 4\n3 5\n4 5\n2 6"
    expected_output = "18"
    run_pie_test_case("../p03787.py", input_content, expected_output)


def test_problem_p03787_2():
    input_content = "3 1\n1 2"
    expected_output = "7"
    run_pie_test_case("../p03787.py", input_content, expected_output)
