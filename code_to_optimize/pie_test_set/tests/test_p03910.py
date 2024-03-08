from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03910_0():
    input_content = "4"
    expected_output = "1\n3"
    run_pie_test_case("../p03910.py", input_content, expected_output)


def test_problem_p03910_1():
    input_content = "4"
    expected_output = "1\n3"
    run_pie_test_case("../p03910.py", input_content, expected_output)


def test_problem_p03910_2():
    input_content = "7"
    expected_output = "1\n2\n4"
    run_pie_test_case("../p03910.py", input_content, expected_output)


def test_problem_p03910_3():
    input_content = "1"
    expected_output = "1"
    run_pie_test_case("../p03910.py", input_content, expected_output)
