from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02556_0():
    input_content = "3\n1 1\n2 4\n3 2"
    expected_output = "4"
    run_pie_test_case("../p02556.py", input_content, expected_output)


def test_problem_p02556_1():
    input_content = "3\n1 1\n2 4\n3 2"
    expected_output = "4"
    run_pie_test_case("../p02556.py", input_content, expected_output)


def test_problem_p02556_2():
    input_content = "2\n1 1\n1 1"
    expected_output = "0"
    run_pie_test_case("../p02556.py", input_content, expected_output)
