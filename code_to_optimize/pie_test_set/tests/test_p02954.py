from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02954_0():
    input_content = "RRLRL"
    expected_output = "0 1 2 1 1"
    run_pie_test_case("../p02954.py", input_content, expected_output)


def test_problem_p02954_1():
    input_content = "RRLLLLRLRRLL"
    expected_output = "0 3 3 0 0 0 1 1 0 2 2 0"
    run_pie_test_case("../p02954.py", input_content, expected_output)


def test_problem_p02954_2():
    input_content = "RRRLLRLLRRRLLLLL"
    expected_output = "0 0 3 2 0 2 1 0 0 0 4 4 0 0 0 0"
    run_pie_test_case("../p02954.py", input_content, expected_output)


def test_problem_p02954_3():
    input_content = "RRLRL"
    expected_output = "0 1 2 1 1"
    run_pie_test_case("../p02954.py", input_content, expected_output)
