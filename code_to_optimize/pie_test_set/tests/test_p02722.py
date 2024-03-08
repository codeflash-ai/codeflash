from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02722_0():
    input_content = "6"
    expected_output = "3"
    run_pie_test_case("../p02722.py", input_content, expected_output)


def test_problem_p02722_1():
    input_content = "314159265358"
    expected_output = "9"
    run_pie_test_case("../p02722.py", input_content, expected_output)


def test_problem_p02722_2():
    input_content = "6"
    expected_output = "3"
    run_pie_test_case("../p02722.py", input_content, expected_output)


def test_problem_p02722_3():
    input_content = "3141"
    expected_output = "13"
    run_pie_test_case("../p02722.py", input_content, expected_output)
