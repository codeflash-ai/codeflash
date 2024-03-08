from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03828_0():
    input_content = "3"
    expected_output = "4"
    run_pie_test_case("../p03828.py", input_content, expected_output)


def test_problem_p03828_1():
    input_content = "6"
    expected_output = "30"
    run_pie_test_case("../p03828.py", input_content, expected_output)


def test_problem_p03828_2():
    input_content = "3"
    expected_output = "4"
    run_pie_test_case("../p03828.py", input_content, expected_output)


def test_problem_p03828_3():
    input_content = "1000"
    expected_output = "972926972"
    run_pie_test_case("../p03828.py", input_content, expected_output)
