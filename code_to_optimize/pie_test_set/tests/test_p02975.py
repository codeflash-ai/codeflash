from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02975_0():
    input_content = "3\n1 2 3"
    expected_output = "Yes"
    run_pie_test_case("../p02975.py", input_content, expected_output)


def test_problem_p02975_1():
    input_content = "3\n1 2 3"
    expected_output = "Yes"
    run_pie_test_case("../p02975.py", input_content, expected_output)


def test_problem_p02975_2():
    input_content = "4\n1 2 4 8"
    expected_output = "No"
    run_pie_test_case("../p02975.py", input_content, expected_output)
