from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00748_0():
    input_content = "40\n14\n5\n165\n120\n103\n106\n139\n0"
    expected_output = "2 6\n2 14\n2 5\n1 1\n1 18\n5 35\n4 4\n3 37"
    run_pie_test_case("../p00748.py", input_content, expected_output)


def test_problem_p00748_1():
    input_content = "40\n14\n5\n165\n120\n103\n106\n139\n0"
    expected_output = "2 6\n2 14\n2 5\n1 1\n1 18\n5 35\n4 4\n3 37"
    run_pie_test_case("../p00748.py", input_content, expected_output)
