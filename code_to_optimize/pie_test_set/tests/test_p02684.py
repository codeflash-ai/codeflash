from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02684_0():
    input_content = "4 5\n3 2 4 1"
    expected_output = "4"
    run_pie_test_case("../p02684.py", input_content, expected_output)


def test_problem_p02684_1():
    input_content = "6 727202214173249351\n6 5 2 5 3 2"
    expected_output = "2"
    run_pie_test_case("../p02684.py", input_content, expected_output)


def test_problem_p02684_2():
    input_content = "4 5\n3 2 4 1"
    expected_output = "4"
    run_pie_test_case("../p02684.py", input_content, expected_output)
