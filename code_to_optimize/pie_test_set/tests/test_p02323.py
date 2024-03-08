from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02323_0():
    input_content = "4 6\n0 1 2\n1 2 3\n1 3 9\n2 0 1\n2 3 6\n3 2 4"
    expected_output = "16"
    run_pie_test_case("../p02323.py", input_content, expected_output)


def test_problem_p02323_1():
    input_content = "3 3\n0 1 1\n1 2 1\n0 2 1"
    expected_output = "-1"
    run_pie_test_case("../p02323.py", input_content, expected_output)


def test_problem_p02323_2():
    input_content = "4 6\n0 1 2\n1 2 3\n1 3 9\n2 0 1\n2 3 6\n3 2 4"
    expected_output = "16"
    run_pie_test_case("../p02323.py", input_content, expected_output)
