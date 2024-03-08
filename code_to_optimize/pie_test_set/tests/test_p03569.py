from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03569_0():
    input_content = "xabxa"
    expected_output = "2"
    run_pie_test_case("../p03569.py", input_content, expected_output)


def test_problem_p03569_1():
    input_content = "a"
    expected_output = "0"
    run_pie_test_case("../p03569.py", input_content, expected_output)


def test_problem_p03569_2():
    input_content = "xabxa"
    expected_output = "2"
    run_pie_test_case("../p03569.py", input_content, expected_output)


def test_problem_p03569_3():
    input_content = "ab"
    expected_output = "-1"
    run_pie_test_case("../p03569.py", input_content, expected_output)


def test_problem_p03569_4():
    input_content = "oxxx"
    expected_output = "3"
    run_pie_test_case("../p03569.py", input_content, expected_output)
