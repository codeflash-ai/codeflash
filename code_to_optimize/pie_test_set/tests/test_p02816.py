from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02816_0():
    input_content = "3\n0 2 1\n1 2 3"
    expected_output = "1 3"
    run_pie_test_case("../p02816.py", input_content, expected_output)


def test_problem_p02816_1():
    input_content = "6\n0 1 3 7 6 4\n1 5 4 6 2 3"
    expected_output = "2 2\n5 5"
    run_pie_test_case("../p02816.py", input_content, expected_output)


def test_problem_p02816_2():
    input_content = "3\n0 2 1\n1 2 3"
    expected_output = "1 3"
    run_pie_test_case("../p02816.py", input_content, expected_output)


def test_problem_p02816_3():
    input_content = "5\n0 0 0 0 0\n2 2 2 2 2"
    expected_output = "0 2\n1 2\n2 2\n3 2\n4 2"
    run_pie_test_case("../p02816.py", input_content, expected_output)
