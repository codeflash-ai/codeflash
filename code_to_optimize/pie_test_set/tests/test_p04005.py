from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04005_0():
    input_content = "3 3 3"
    expected_output = "9"
    run_pie_test_case("../p04005.py", input_content, expected_output)


def test_problem_p04005_1():
    input_content = "2 2 4"
    expected_output = "0"
    run_pie_test_case("../p04005.py", input_content, expected_output)


def test_problem_p04005_2():
    input_content = "3 3 3"
    expected_output = "9"
    run_pie_test_case("../p04005.py", input_content, expected_output)


def test_problem_p04005_3():
    input_content = "5 3 5"
    expected_output = "15"
    run_pie_test_case("../p04005.py", input_content, expected_output)
