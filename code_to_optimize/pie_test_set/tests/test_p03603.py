from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03603_0():
    input_content = "3\n1 1\n4 3 2"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03603.py", input_content, expected_output)


def test_problem_p03603_1():
    input_content = "3\n1 1\n4 3 2"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03603.py", input_content, expected_output)


def test_problem_p03603_2():
    input_content = "8\n1 1 1 3 4 5 5\n4 1 6 2 2 1 3 3"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03603.py", input_content, expected_output)


def test_problem_p03603_3():
    input_content = "1\n\n0"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03603.py", input_content, expected_output)


def test_problem_p03603_4():
    input_content = "3\n1 2\n1 2 3"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p03603.py", input_content, expected_output)
