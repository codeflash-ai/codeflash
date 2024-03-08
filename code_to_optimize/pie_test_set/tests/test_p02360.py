from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02360_0():
    input_content = "2\n0 0 3 2\n2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p02360.py", input_content, expected_output)


def test_problem_p02360_1():
    input_content = "2\n0 0 2 2\n2 0 4 2"
    expected_output = "1"
    run_pie_test_case("../p02360.py", input_content, expected_output)


def test_problem_p02360_2():
    input_content = "3\n0 0 2 2\n0 0 2 2\n0 0 2 2"
    expected_output = "3"
    run_pie_test_case("../p02360.py", input_content, expected_output)


def test_problem_p02360_3():
    input_content = "2\n0 0 3 2\n2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p02360.py", input_content, expected_output)
