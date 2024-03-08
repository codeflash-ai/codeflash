from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02408_0():
    input_content = "47\nS 10\nS 11\nS 12\nS 13\nH 1\nH 2\nS 6\nS 7\nS 8\nS 9\nH 6\nH 8\nH 9\nH 10\nH 11\nH 4\nH 5\nS 2\nS 3\nS 4\nS 5\nH 12\nH 13\nC 1\nC 2\nD 1\nD 2\nD 3\nD 4\nD 5\nD 6\nD 7\nC 3\nC 4\nC 5\nC 6\nC 7\nC 8\nC 9\nC 10\nC 11\nC 13\nD 9\nD 10\nD 11\nD 12\nD 13"
    expected_output = "S 1\nH 3\nH 7\nC 12\nD 8"
    run_pie_test_case("../p02408.py", input_content, expected_output)


def test_problem_p02408_1():
    input_content = "47\nS 10\nS 11\nS 12\nS 13\nH 1\nH 2\nS 6\nS 7\nS 8\nS 9\nH 6\nH 8\nH 9\nH 10\nH 11\nH 4\nH 5\nS 2\nS 3\nS 4\nS 5\nH 12\nH 13\nC 1\nC 2\nD 1\nD 2\nD 3\nD 4\nD 5\nD 6\nD 7\nC 3\nC 4\nC 5\nC 6\nC 7\nC 8\nC 9\nC 10\nC 11\nC 13\nD 9\nD 10\nD 11\nD 12\nD 13"
    expected_output = "S 1\nH 3\nH 7\nC 12\nD 8"
    run_pie_test_case("../p02408.py", input_content, expected_output)
