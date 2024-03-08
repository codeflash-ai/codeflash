from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03330_0():
    input_content = "2 3\n0 1 1\n1 0 1\n1 4 0\n1 2\n3 3"
    expected_output = "3"
    run_pie_test_case("../p03330.py", input_content, expected_output)


def test_problem_p03330_1():
    input_content = "4 3\n0 12 71\n81 0 53\n14 92 0\n1 1 2 1\n2 1 1 2\n2 2 1 3\n1 1 2 2"
    expected_output = "428"
    run_pie_test_case("../p03330.py", input_content, expected_output)


def test_problem_p03330_2():
    input_content = "2 3\n0 1 1\n1 0 1\n1 4 0\n1 2\n3 3"
    expected_output = "3"
    run_pie_test_case("../p03330.py", input_content, expected_output)
