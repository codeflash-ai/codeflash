from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02957_0():
    input_content = "2 16"
    expected_output = "9"
    run_pie_test_case("../p02957.py", input_content, expected_output)


def test_problem_p02957_1():
    input_content = "998244353 99824435"
    expected_output = "549034394"
    run_pie_test_case("../p02957.py", input_content, expected_output)


def test_problem_p02957_2():
    input_content = "0 3"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p02957.py", input_content, expected_output)


def test_problem_p02957_3():
    input_content = "2 16"
    expected_output = "9"
    run_pie_test_case("../p02957.py", input_content, expected_output)
