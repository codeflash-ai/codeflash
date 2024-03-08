from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03862_0():
    input_content = "3 3\n2 2 2"
    expected_output = "1"
    run_pie_test_case("../p03862.py", input_content, expected_output)


def test_problem_p03862_1():
    input_content = "2 0\n5 5"
    expected_output = "10"
    run_pie_test_case("../p03862.py", input_content, expected_output)


def test_problem_p03862_2():
    input_content = "5 9\n3 1 4 1 5"
    expected_output = "0"
    run_pie_test_case("../p03862.py", input_content, expected_output)


def test_problem_p03862_3():
    input_content = "3 3\n2 2 2"
    expected_output = "1"
    run_pie_test_case("../p03862.py", input_content, expected_output)


def test_problem_p03862_4():
    input_content = "6 1\n1 6 1 2 0 4"
    expected_output = "11"
    run_pie_test_case("../p03862.py", input_content, expected_output)
