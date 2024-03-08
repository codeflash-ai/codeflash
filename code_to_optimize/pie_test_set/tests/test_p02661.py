from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02661_0():
    input_content = "2\n1 2\n2 3"
    expected_output = "3"
    run_pie_test_case("../p02661.py", input_content, expected_output)


def test_problem_p02661_1():
    input_content = "3\n100 100\n10 10000\n1 1000000000"
    expected_output = "9991"
    run_pie_test_case("../p02661.py", input_content, expected_output)


def test_problem_p02661_2():
    input_content = "2\n1 2\n2 3"
    expected_output = "3"
    run_pie_test_case("../p02661.py", input_content, expected_output)
