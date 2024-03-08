from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02560_0():
    input_content = "5\n4 10 6 3\n6 5 4 3\n1 1 0 0\n31415 92653 58979 32384\n1000000000 1000000000 999999999 999999999"
    expected_output = "3\n13\n0\n314095480\n499999999500000000"
    run_pie_test_case("../p02560.py", input_content, expected_output)


def test_problem_p02560_1():
    input_content = "5\n4 10 6 3\n6 5 4 3\n1 1 0 0\n31415 92653 58979 32384\n1000000000 1000000000 999999999 999999999"
    expected_output = "3\n13\n0\n314095480\n499999999500000000"
    run_pie_test_case("../p02560.py", input_content, expected_output)
