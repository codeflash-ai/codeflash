from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02856_0():
    input_content = "2\n2 2\n9 1"
    expected_output = "3"
    run_pie_test_case("../p02856.py", input_content, expected_output)


def test_problem_p02856_1():
    input_content = "3\n1 1\n0 8\n7 1"
    expected_output = "9"
    run_pie_test_case("../p02856.py", input_content, expected_output)


def test_problem_p02856_2():
    input_content = "2\n2 2\n9 1"
    expected_output = "3"
    run_pie_test_case("../p02856.py", input_content, expected_output)
