from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03441_0():
    input_content = "5\n0 1\n0 2\n0 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03441.py", input_content, expected_output)


def test_problem_p03441_1():
    input_content = "2\n0 1"
    expected_output = "1"
    run_pie_test_case("../p03441.py", input_content, expected_output)


def test_problem_p03441_2():
    input_content = "5\n0 1\n0 2\n0 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03441.py", input_content, expected_output)


def test_problem_p03441_3():
    input_content = "10\n2 8\n6 0\n4 1\n7 6\n2 3\n8 6\n6 9\n2 4\n5 8"
    expected_output = "3"
    run_pie_test_case("../p03441.py", input_content, expected_output)
