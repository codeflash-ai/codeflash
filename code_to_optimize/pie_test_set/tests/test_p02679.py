from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02679_0():
    input_content = "3\n1 2\n-1 1\n2 -1"
    expected_output = "5"
    run_pie_test_case("../p02679.py", input_content, expected_output)


def test_problem_p02679_1():
    input_content = "3\n1 2\n-1 1\n2 -1"
    expected_output = "5"
    run_pie_test_case("../p02679.py", input_content, expected_output)


def test_problem_p02679_2():
    input_content = "10\n3 2\n3 2\n-1 1\n2 -1\n-3 -9\n-8 12\n7 7\n8 1\n8 2\n8 4"
    expected_output = "479"
    run_pie_test_case("../p02679.py", input_content, expected_output)
