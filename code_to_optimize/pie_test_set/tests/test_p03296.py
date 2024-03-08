from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03296_0():
    input_content = "5\n1 1 2 2 2"
    expected_output = "2"
    run_pie_test_case("../p03296.py", input_content, expected_output)


def test_problem_p03296_1():
    input_content = "5\n1 1 2 2 2"
    expected_output = "2"
    run_pie_test_case("../p03296.py", input_content, expected_output)


def test_problem_p03296_2():
    input_content = "3\n1 2 1"
    expected_output = "0"
    run_pie_test_case("../p03296.py", input_content, expected_output)


def test_problem_p03296_3():
    input_content = "14\n1 2 2 3 3 3 4 4 4 4 1 2 3 4"
    expected_output = "4"
    run_pie_test_case("../p03296.py", input_content, expected_output)


def test_problem_p03296_4():
    input_content = "5\n1 1 1 1 1"
    expected_output = "2"
    run_pie_test_case("../p03296.py", input_content, expected_output)
