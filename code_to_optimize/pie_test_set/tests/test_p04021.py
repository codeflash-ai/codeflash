from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04021_0():
    input_content = "4\n2\n4\n3\n1"
    expected_output = "1"
    run_pie_test_case("../p04021.py", input_content, expected_output)


def test_problem_p04021_1():
    input_content = "5\n10\n8\n5\n3\n2"
    expected_output = "0"
    run_pie_test_case("../p04021.py", input_content, expected_output)


def test_problem_p04021_2():
    input_content = "4\n2\n4\n3\n1"
    expected_output = "1"
    run_pie_test_case("../p04021.py", input_content, expected_output)
