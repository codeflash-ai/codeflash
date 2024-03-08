from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02977_0():
    input_content = "3"
    expected_output = "Yes\n1 2\n2 3\n3 4\n4 5\n5 6"
    run_pie_test_case("../p02977.py", input_content, expected_output)


def test_problem_p02977_1():
    input_content = "1"
    expected_output = "No"
    run_pie_test_case("../p02977.py", input_content, expected_output)


def test_problem_p02977_2():
    input_content = "3"
    expected_output = "Yes\n1 2\n2 3\n3 4\n4 5\n5 6"
    run_pie_test_case("../p02977.py", input_content, expected_output)
