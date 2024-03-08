from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03458_0():
    input_content = "4 3\n0 1 W\n1 2 W\n5 3 B\n5 4 B"
    expected_output = "4"
    run_pie_test_case("../p03458.py", input_content, expected_output)


def test_problem_p03458_1():
    input_content = "6 2\n1 2 B\n2 1 W\n2 2 B\n1 0 B\n0 6 W\n4 5 W"
    expected_output = "4"
    run_pie_test_case("../p03458.py", input_content, expected_output)


def test_problem_p03458_2():
    input_content = "4 3\n0 1 W\n1 2 W\n5 3 B\n5 4 B"
    expected_output = "4"
    run_pie_test_case("../p03458.py", input_content, expected_output)


def test_problem_p03458_3():
    input_content = "2 1000\n0 0 B\n0 1 W"
    expected_output = "2"
    run_pie_test_case("../p03458.py", input_content, expected_output)
