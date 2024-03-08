from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04008_0():
    input_content = "3 1\n2 3 1"
    expected_output = "2"
    run_pie_test_case("../p04008.py", input_content, expected_output)


def test_problem_p04008_1():
    input_content = "8 2\n4 1 2 3 1 2 3 4"
    expected_output = "3"
    run_pie_test_case("../p04008.py", input_content, expected_output)


def test_problem_p04008_2():
    input_content = "3 1\n2 3 1"
    expected_output = "2"
    run_pie_test_case("../p04008.py", input_content, expected_output)


def test_problem_p04008_3():
    input_content = "4 2\n1 1 2 2"
    expected_output = "0"
    run_pie_test_case("../p04008.py", input_content, expected_output)
