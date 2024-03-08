from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00037_0():
    input_content = "1111\n00001\n0110\n01011\n0010\n01111\n0010\n01001\n0111"
    expected_output = "RRRRDDDDLLLUUURRDDLURULLDDDRRRUUUULLLL"
    run_pie_test_case("../p00037.py", input_content, expected_output)


def test_problem_p00037_1():
    input_content = "1111\n00001\n0110\n01011\n0010\n01111\n0010\n01001\n0111"
    expected_output = "RRRRDDDDLLLUUURRDDLURULLDDDRRRUUUULLLL"
    run_pie_test_case("../p00037.py", input_content, expected_output)
