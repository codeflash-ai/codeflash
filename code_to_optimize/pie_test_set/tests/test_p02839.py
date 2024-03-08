from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02839_0():
    input_content = "2 2\n1 2\n3 4\n3 4\n2 1"
    expected_output = "0"
    run_pie_test_case("../p02839.py", input_content, expected_output)


def test_problem_p02839_1():
    input_content = "2 2\n1 2\n3 4\n3 4\n2 1"
    expected_output = "0"
    run_pie_test_case("../p02839.py", input_content, expected_output)


def test_problem_p02839_2():
    input_content = "2 3\n1 10 80\n80 10 1\n1 2 3\n4 5 6"
    expected_output = "2"
    run_pie_test_case("../p02839.py", input_content, expected_output)
