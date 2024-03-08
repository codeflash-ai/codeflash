from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02238_0():
    input_content = "4\n1 1 2\n2 1 4\n3 0\n4 1 3"
    expected_output = "1 1 8\n2 2 7\n3 4 5\n4 3 6"
    run_pie_test_case("../p02238.py", input_content, expected_output)


def test_problem_p02238_1():
    input_content = "4\n1 1 2\n2 1 4\n3 0\n4 1 3"
    expected_output = "1 1 8\n2 2 7\n3 4 5\n4 3 6"
    run_pie_test_case("../p02238.py", input_content, expected_output)


def test_problem_p02238_2():
    input_content = "6\n1 2 2 3\n2 2 3 4\n3 1 5\n4 1 6\n5 1 6\n6 0"
    expected_output = "1 1 12\n2 2 11\n3 3 8\n4 9 10\n5 4 7\n6 5 6"
    run_pie_test_case("../p02238.py", input_content, expected_output)
