from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02397_0():
    input_content = "3 2\n2 2\n5 3\n0 0"
    expected_output = "2 3\n2 2\n3 5"
    run_pie_test_case("../p02397.py", input_content, expected_output)


def test_problem_p02397_1():
    input_content = "3 2\n2 2\n5 3\n0 0"
    expected_output = "2 3\n2 2\n3 5"
    run_pie_test_case("../p02397.py", input_content, expected_output)
