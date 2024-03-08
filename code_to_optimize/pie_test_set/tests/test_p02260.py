from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02260_0():
    input_content = "6\n5 6 4 2 1 3"
    expected_output = "1 2 3 4 5 6\n4"
    run_pie_test_case("../p02260.py", input_content, expected_output)


def test_problem_p02260_1():
    input_content = "6\n5 2 4 6 1 3"
    expected_output = "1 2 3 4 5 6\n3"
    run_pie_test_case("../p02260.py", input_content, expected_output)


def test_problem_p02260_2():
    input_content = "6\n5 6 4 2 1 3"
    expected_output = "1 2 3 4 5 6\n4"
    run_pie_test_case("../p02260.py", input_content, expected_output)
