from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02371_0():
    input_content = "4\n0 1 2\n1 2 1\n1 3 3"
    expected_output = "5"
    run_pie_test_case("../p02371.py", input_content, expected_output)


def test_problem_p02371_1():
    input_content = "4\n0 1 1\n1 2 2\n2 3 4"
    expected_output = "7"
    run_pie_test_case("../p02371.py", input_content, expected_output)


def test_problem_p02371_2():
    input_content = "4\n0 1 2\n1 2 1\n1 3 3"
    expected_output = "5"
    run_pie_test_case("../p02371.py", input_content, expected_output)
