from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04004_0():
    input_content = "1 1 1"
    expected_output = "17"
    run_pie_test_case("../p04004.py", input_content, expected_output)


def test_problem_p04004_1():
    input_content = "4 2 2"
    expected_output = "1227"
    run_pie_test_case("../p04004.py", input_content, expected_output)


def test_problem_p04004_2():
    input_content = "1 1 1"
    expected_output = "17"
    run_pie_test_case("../p04004.py", input_content, expected_output)


def test_problem_p04004_3():
    input_content = "1000 1000 1000"
    expected_output = "261790852"
    run_pie_test_case("../p04004.py", input_content, expected_output)
