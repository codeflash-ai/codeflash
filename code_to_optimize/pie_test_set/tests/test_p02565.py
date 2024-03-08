from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02565_0():
    input_content = "3 2\n1 4\n2 5\n0 6"
    expected_output = "Yes\n4\n2\n0"
    run_pie_test_case("../p02565.py", input_content, expected_output)


def test_problem_p02565_1():
    input_content = "3 3\n1 4\n2 5\n0 6"
    expected_output = "No"
    run_pie_test_case("../p02565.py", input_content, expected_output)


def test_problem_p02565_2():
    input_content = "3 2\n1 4\n2 5\n0 6"
    expected_output = "Yes\n4\n2\n0"
    run_pie_test_case("../p02565.py", input_content, expected_output)
