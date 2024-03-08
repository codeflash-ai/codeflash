from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02787_0():
    input_content = "9 3\n8 3\n4 2\n2 1"
    expected_output = "4"
    run_pie_test_case("../p02787.py", input_content, expected_output)


def test_problem_p02787_1():
    input_content = "9 3\n8 3\n4 2\n2 1"
    expected_output = "4"
    run_pie_test_case("../p02787.py", input_content, expected_output)


def test_problem_p02787_2():
    input_content = "9999 10\n540 7550\n691 9680\n700 9790\n510 7150\n415 5818\n551 7712\n587 8227\n619 8671\n588 8228\n176 2461"
    expected_output = "139815"
    run_pie_test_case("../p02787.py", input_content, expected_output)


def test_problem_p02787_3():
    input_content = "100 6\n1 1\n2 3\n3 9\n4 27\n5 81\n6 243"
    expected_output = "100"
    run_pie_test_case("../p02787.py", input_content, expected_output)
