from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02329_0():
    input_content = "3 14\n3 1 2\n4 8 2\n1 2 3\n7 3 2"
    expected_output = "9"
    run_pie_test_case("../p02329.py", input_content, expected_output)


def test_problem_p02329_1():
    input_content = "3 14\n3 1 2\n4 8 2\n1 2 3\n7 3 2"
    expected_output = "9"
    run_pie_test_case("../p02329.py", input_content, expected_output)


def test_problem_p02329_2():
    input_content = "5 4\n1 1 1 1 1\n1 1 1 1 1\n1 1 1 1 1\n1 1 1 1 1"
    expected_output = "625"
    run_pie_test_case("../p02329.py", input_content, expected_output)
