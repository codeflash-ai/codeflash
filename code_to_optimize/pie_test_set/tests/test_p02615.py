from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02615_0():
    input_content = "4\n2 2 1 3"
    expected_output = "7"
    run_pie_test_case("../p02615.py", input_content, expected_output)


def test_problem_p02615_1():
    input_content = "4\n2 2 1 3"
    expected_output = "7"
    run_pie_test_case("../p02615.py", input_content, expected_output)


def test_problem_p02615_2():
    input_content = "7\n1 1 1 1 1 1 1"
    expected_output = "6"
    run_pie_test_case("../p02615.py", input_content, expected_output)
