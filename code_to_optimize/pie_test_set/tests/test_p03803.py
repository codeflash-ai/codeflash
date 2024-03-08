from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03803_0():
    input_content = "8 6"
    expected_output = "Alice"
    run_pie_test_case("../p03803.py", input_content, expected_output)


def test_problem_p03803_1():
    input_content = "8 6"
    expected_output = "Alice"
    run_pie_test_case("../p03803.py", input_content, expected_output)


def test_problem_p03803_2():
    input_content = "1 1"
    expected_output = "Draw"
    run_pie_test_case("../p03803.py", input_content, expected_output)


def test_problem_p03803_3():
    input_content = "13 1"
    expected_output = "Bob"
    run_pie_test_case("../p03803.py", input_content, expected_output)
