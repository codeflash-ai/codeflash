from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02377_0():
    input_content = "4 5 2\n0 1 2 1\n0 2 1 2\n1 2 1 1\n1 3 1 3\n2 3 2 1"
    expected_output = "6"
    run_pie_test_case("../p02377.py", input_content, expected_output)


def test_problem_p02377_1():
    input_content = "4 5 2\n0 1 2 1\n0 2 1 2\n1 2 1 1\n1 3 1 3\n2 3 2 1"
    expected_output = "6"
    run_pie_test_case("../p02377.py", input_content, expected_output)


def test_problem_p02377_2():
    input_content = ""
    expected_output = ""
    run_pie_test_case("../p02377.py", input_content, expected_output)
