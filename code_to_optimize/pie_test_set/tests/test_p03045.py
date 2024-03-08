from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03045_0():
    input_content = "3 1\n1 2 1"
    expected_output = "2"
    run_pie_test_case("../p03045.py", input_content, expected_output)


def test_problem_p03045_1():
    input_content = "100000 1\n1 100000 100"
    expected_output = "99999"
    run_pie_test_case("../p03045.py", input_content, expected_output)


def test_problem_p03045_2():
    input_content = "6 5\n1 2 1\n2 3 2\n1 3 3\n4 5 4\n5 6 5"
    expected_output = "2"
    run_pie_test_case("../p03045.py", input_content, expected_output)


def test_problem_p03045_3():
    input_content = "3 1\n1 2 1"
    expected_output = "2"
    run_pie_test_case("../p03045.py", input_content, expected_output)
