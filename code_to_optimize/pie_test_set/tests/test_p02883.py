from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02883_0():
    input_content = "3 5\n4 2 1\n2 3 1"
    expected_output = "2"
    run_pie_test_case("../p02883.py", input_content, expected_output)


def test_problem_p02883_1():
    input_content = "3 8\n4 2 1\n2 3 1"
    expected_output = "0"
    run_pie_test_case("../p02883.py", input_content, expected_output)


def test_problem_p02883_2():
    input_content = "3 5\n4 2 1\n2 3 1"
    expected_output = "2"
    run_pie_test_case("../p02883.py", input_content, expected_output)


def test_problem_p02883_3():
    input_content = "11 14\n3 1 4 1 5 9 2 6 5 3 5\n8 9 7 9 3 2 3 8 4 6 2"
    expected_output = "12"
    run_pie_test_case("../p02883.py", input_content, expected_output)
