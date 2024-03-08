from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02626_0():
    input_content = "2\n5 3"
    expected_output = "1"
    run_pie_test_case("../p02626.py", input_content, expected_output)


def test_problem_p02626_1():
    input_content = "2\n3 5"
    expected_output = "-1"
    run_pie_test_case("../p02626.py", input_content, expected_output)


def test_problem_p02626_2():
    input_content = "3\n4294967297 8589934593 12884901890"
    expected_output = "1"
    run_pie_test_case("../p02626.py", input_content, expected_output)


def test_problem_p02626_3():
    input_content = "8\n10 9 8 7 6 5 4 3"
    expected_output = "3"
    run_pie_test_case("../p02626.py", input_content, expected_output)


def test_problem_p02626_4():
    input_content = "2\n5 3"
    expected_output = "1"
    run_pie_test_case("../p02626.py", input_content, expected_output)


def test_problem_p02626_5():
    input_content = "3\n1 1 2"
    expected_output = "-1"
    run_pie_test_case("../p02626.py", input_content, expected_output)
