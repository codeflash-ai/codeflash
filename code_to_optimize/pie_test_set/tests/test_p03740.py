from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03740_0():
    input_content = "2 1"
    expected_output = "Brown"
    run_pie_test_case("../p03740.py", input_content, expected_output)


def test_problem_p03740_1():
    input_content = "0 0"
    expected_output = "Brown"
    run_pie_test_case("../p03740.py", input_content, expected_output)


def test_problem_p03740_2():
    input_content = "5 0"
    expected_output = "Alice"
    run_pie_test_case("../p03740.py", input_content, expected_output)


def test_problem_p03740_3():
    input_content = "2 1"
    expected_output = "Brown"
    run_pie_test_case("../p03740.py", input_content, expected_output)


def test_problem_p03740_4():
    input_content = "4 8"
    expected_output = "Alice"
    run_pie_test_case("../p03740.py", input_content, expected_output)
