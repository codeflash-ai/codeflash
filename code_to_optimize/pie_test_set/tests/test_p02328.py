from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02328_0():
    input_content = "8\n2 1 3 5 3 4 2 1"
    expected_output = "12"
    run_pie_test_case("../p02328.py", input_content, expected_output)


def test_problem_p02328_1():
    input_content = "8\n2 1 3 5 3 4 2 1"
    expected_output = "12"
    run_pie_test_case("../p02328.py", input_content, expected_output)


def test_problem_p02328_2():
    input_content = "3\n2 0 1"
    expected_output = "2"
    run_pie_test_case("../p02328.py", input_content, expected_output)
