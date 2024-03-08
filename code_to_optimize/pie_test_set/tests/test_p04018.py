from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04018_0():
    input_content = "aab"
    expected_output = "1\n1"
    run_pie_test_case("../p04018.py", input_content, expected_output)


def test_problem_p04018_1():
    input_content = "ddd"
    expected_output = "3\n1"
    run_pie_test_case("../p04018.py", input_content, expected_output)


def test_problem_p04018_2():
    input_content = "aab"
    expected_output = "1\n1"
    run_pie_test_case("../p04018.py", input_content, expected_output)


def test_problem_p04018_3():
    input_content = "bcbc"
    expected_output = "2\n3"
    run_pie_test_case("../p04018.py", input_content, expected_output)
