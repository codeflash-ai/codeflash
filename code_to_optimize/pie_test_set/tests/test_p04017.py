from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04017_0():
    input_content = "9\n1 3 6 13 15 18 19 29 31\n10\n4\n1 8\n7 3\n6 7\n8 5"
    expected_output = "4\n2\n1\n2"
    run_pie_test_case("../p04017.py", input_content, expected_output)


def test_problem_p04017_1():
    input_content = "9\n1 3 6 13 15 18 19 29 31\n10\n4\n1 8\n7 3\n6 7\n8 5"
    expected_output = "4\n2\n1\n2"
    run_pie_test_case("../p04017.py", input_content, expected_output)
