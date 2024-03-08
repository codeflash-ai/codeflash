from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02320_0():
    input_content = "4 8\n4 3 2\n2 1 1\n1 2 4\n3 2 2"
    expected_output = "12"
    run_pie_test_case("../p02320.py", input_content, expected_output)


def test_problem_p02320_1():
    input_content = "2 100\n1 1 100\n2 1 50"
    expected_output = "150"
    run_pie_test_case("../p02320.py", input_content, expected_output)


def test_problem_p02320_2():
    input_content = "4 8\n4 3 2\n2 1 1\n1 2 4\n3 2 2"
    expected_output = "12"
    run_pie_test_case("../p02320.py", input_content, expected_output)
