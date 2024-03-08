from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00496_0():
    input_content = "5 20 14\n8 9\n2 4\n7 13\n6 3\n5 8"
    expected_output = "16"
    run_pie_test_case("../p00496.py", input_content, expected_output)


def test_problem_p00496_1():
    input_content = "5 20 14\n8 9\n2 4\n7 13\n6 3\n5 8"
    expected_output = "16"
    run_pie_test_case("../p00496.py", input_content, expected_output)


def test_problem_p00496_2():
    input_content = "None"
    expected_output = "None"
    run_pie_test_case("../p00496.py", input_content, expected_output)
