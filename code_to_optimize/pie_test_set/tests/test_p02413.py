from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02413_0():
    input_content = "4 5\n1 1 3 4 5\n2 2 2 4 5\n3 3 0 1 1\n2 3 4 4 6"
    expected_output = "1 1 3 4 5 14\n2 2 2 4 5 15\n3 3 0 1 1 8\n2 3 4 4 6 19\n8 9 9 13 17 56"
    run_pie_test_case("../p02413.py", input_content, expected_output)


def test_problem_p02413_1():
    input_content = "4 5\n1 1 3 4 5\n2 2 2 4 5\n3 3 0 1 1\n2 3 4 4 6"
    expected_output = "1 1 3 4 5 14\n2 2 2 4 5 15\n3 3 0 1 1 8\n2 3 4 4 6 19\n8 9 9 13 17 56"
    run_pie_test_case("../p02413.py", input_content, expected_output)
