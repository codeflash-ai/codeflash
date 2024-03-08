from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02116_0():
    input_content = "2"
    expected_output = "1"
    run_pie_test_case("../p02116.py", input_content, expected_output)


def test_problem_p02116_1():
    input_content = "3"
    expected_output = "4"
    run_pie_test_case("../p02116.py", input_content, expected_output)


def test_problem_p02116_2():
    input_content = "2"
    expected_output = "1"
    run_pie_test_case("../p02116.py", input_content, expected_output)


def test_problem_p02116_3():
    input_content = "111"
    expected_output = "16"
    run_pie_test_case("../p02116.py", input_content, expected_output)
