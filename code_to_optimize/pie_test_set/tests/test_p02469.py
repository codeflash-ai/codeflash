from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02469_0():
    input_content = "3\n3 4 6"
    expected_output = "12"
    run_pie_test_case("../p02469.py", input_content, expected_output)


def test_problem_p02469_1():
    input_content = "4\n1 2 3 5"
    expected_output = "30"
    run_pie_test_case("../p02469.py", input_content, expected_output)


def test_problem_p02469_2():
    input_content = "3\n3 4 6"
    expected_output = "12"
    run_pie_test_case("../p02469.py", input_content, expected_output)
