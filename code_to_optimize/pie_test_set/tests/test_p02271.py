from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02271_0():
    input_content = "5\n1 5 7 10 21\n8\n2 4 17 8 22 21 100 35"
    expected_output = "no\nno\nyes\nyes\nyes\nyes\nno\nno"
    run_pie_test_case("../p02271.py", input_content, expected_output)


def test_problem_p02271_1():
    input_content = "5\n1 5 7 10 21\n8\n2 4 17 8 22 21 100 35"
    expected_output = "no\nno\nyes\nyes\nyes\nyes\nno\nno"
    run_pie_test_case("../p02271.py", input_content, expected_output)
