from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04047_0():
    input_content = "2\n1 3 1 2"
    expected_output = "3"
    run_pie_test_case("../p04047.py", input_content, expected_output)


def test_problem_p04047_1():
    input_content = "5\n100 1 2 3 14 15 58 58 58 29"
    expected_output = "135"
    run_pie_test_case("../p04047.py", input_content, expected_output)


def test_problem_p04047_2():
    input_content = "2\n1 3 1 2"
    expected_output = "3"
    run_pie_test_case("../p04047.py", input_content, expected_output)
