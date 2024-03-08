from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00629_0():
    input_content = "6\n1 1 6 200\n2 1 6 300\n3 1 6 400\n4 2 5 1200\n5 1 5 1400\n6 3 4 800\n3\n777 1 5 300\n808 2 4 20\n123 3 6 500\n2\n2 1 3 100\n1 1 3 100\n0"
    expected_output = "1\n2\n3\n4\n6\n123\n777\n808\n1\n2"
    run_pie_test_case("../p00629.py", input_content, expected_output)


def test_problem_p00629_1():
    input_content = "6\n1 1 6 200\n2 1 6 300\n3 1 6 400\n4 2 5 1200\n5 1 5 1400\n6 3 4 800\n3\n777 1 5 300\n808 2 4 20\n123 3 6 500\n2\n2 1 3 100\n1 1 3 100\n0"
    expected_output = "1\n2\n3\n4\n6\n123\n777\n808\n1\n2"
    run_pie_test_case("../p00629.py", input_content, expected_output)
