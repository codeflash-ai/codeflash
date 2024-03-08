from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00437_0():
    input_content = "2 2 2\n4\n2 4 5 0\n2 3 6 0\n1 4 5 0\n2 3 5 1\n0 0 0"
    expected_output = "2\n1\n1\n0\n1\n0"
    run_pie_test_case("../p00437.py", input_content, expected_output)


def test_problem_p00437_1():
    input_content = "2 2 2\n4\n2 4 5 0\n2 3 6 0\n1 4 5 0\n2 3 5 1\n0 0 0"
    expected_output = "2\n1\n1\n0\n1\n0"
    run_pie_test_case("../p00437.py", input_content, expected_output)
