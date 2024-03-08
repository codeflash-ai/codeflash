from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02761_0():
    input_content = "3 3\n1 7\n3 2\n1 7"
    expected_output = "702"
    run_pie_test_case("../p02761.py", input_content, expected_output)


def test_problem_p02761_1():
    input_content = "3 1\n1 0"
    expected_output = "-1"
    run_pie_test_case("../p02761.py", input_content, expected_output)


def test_problem_p02761_2():
    input_content = "3 3\n1 7\n3 2\n1 7"
    expected_output = "702"
    run_pie_test_case("../p02761.py", input_content, expected_output)


def test_problem_p02761_3():
    input_content = "3 2\n2 1\n2 3"
    expected_output = "-1"
    run_pie_test_case("../p02761.py", input_content, expected_output)
