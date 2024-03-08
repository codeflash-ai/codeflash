from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03497_0():
    input_content = "5 2\n1 1 2 2 5"
    expected_output = "1"
    run_pie_test_case("../p03497.py", input_content, expected_output)


def test_problem_p03497_1():
    input_content = "4 4\n1 1 2 2"
    expected_output = "0"
    run_pie_test_case("../p03497.py", input_content, expected_output)


def test_problem_p03497_2():
    input_content = "5 2\n1 1 2 2 5"
    expected_output = "1"
    run_pie_test_case("../p03497.py", input_content, expected_output)


def test_problem_p03497_3():
    input_content = "10 3\n5 1 3 2 4 1 1 2 3 4"
    expected_output = "3"
    run_pie_test_case("../p03497.py", input_content, expected_output)
