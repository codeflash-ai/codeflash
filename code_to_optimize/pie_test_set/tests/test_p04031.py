from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04031_0():
    input_content = "2\n4 8"
    expected_output = "8"
    run_pie_test_case("../p04031.py", input_content, expected_output)


def test_problem_p04031_1():
    input_content = "2\n4 8"
    expected_output = "8"
    run_pie_test_case("../p04031.py", input_content, expected_output)


def test_problem_p04031_2():
    input_content = "3\n1 1 3"
    expected_output = "3"
    run_pie_test_case("../p04031.py", input_content, expected_output)


def test_problem_p04031_3():
    input_content = "3\n4 2 5"
    expected_output = "5"
    run_pie_test_case("../p04031.py", input_content, expected_output)


def test_problem_p04031_4():
    input_content = "4\n-100 -100 -100 -100"
    expected_output = "0"
    run_pie_test_case("../p04031.py", input_content, expected_output)
