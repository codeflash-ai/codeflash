from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02277_0():
    input_content = "6\nD 3\nH 2\nD 1\nS 3\nD 2\nC 1"
    expected_output = "Not stable\nD 1\nC 1\nD 2\nH 2\nD 3\nS 3"
    run_pie_test_case("../p02277.py", input_content, expected_output)


def test_problem_p02277_1():
    input_content = "2\nS 1\nH 1"
    expected_output = "Stable\nS 1\nH 1"
    run_pie_test_case("../p02277.py", input_content, expected_output)


def test_problem_p02277_2():
    input_content = "6\nD 3\nH 2\nD 1\nS 3\nD 2\nC 1"
    expected_output = "Not stable\nD 1\nC 1\nD 2\nH 2\nD 3\nS 3"
    run_pie_test_case("../p02277.py", input_content, expected_output)
