from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02274_0():
    input_content = "5\n3 5 2 1 4"
    expected_output = "6"
    run_pie_test_case("../p02274.py", input_content, expected_output)


def test_problem_p02274_1():
    input_content = "3\n3 1 2"
    expected_output = "2"
    run_pie_test_case("../p02274.py", input_content, expected_output)


def test_problem_p02274_2():
    input_content = "5\n3 5 2 1 4"
    expected_output = "6"
    run_pie_test_case("../p02274.py", input_content, expected_output)
